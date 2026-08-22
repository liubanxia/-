from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from .db import KnowledgeDB
from .llm import LocalLLM
from .pdf_assets import PDFAssetStore, markdown_images
from .retrieval import Evidence, Retriever


ProgressCallback = Callable[[int, int, str], None]
PauseCallback = Callable[[], bool]
_CITATION_RE = re.compile(r"\[S(\d+)\]")
_RESUMABLE_STATUS = {"queued", "running", "failed", "paused"}
_QUERY_SPLIT_RE = re.compile(r"[\n\r；;。！？!?]+")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


def _generation_budget(prompt: str, *, ceiling: int) -> int:
    """Reserve output in proportion to evidence size, never at a fixed maximum."""

    chars = len(str(prompt or ""))
    return max(512, min(int(ceiling), 360 + int(chars * 0.10)))


class OrganizePaused(RuntimeError):
    def __init__(self, task_id: int):
        super().__init__(f"整理任务已暂停: {task_id}")
        self.task_id = int(task_id)


def _safe_filename(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|\r\n]+", "_", text).strip(" ._")
    return (text or "未命名专题")[:96]


class DeepOrganizer:
    def __init__(
        self,
        db: KnowledgeDB,
        retriever: Retriever,
        llm: LocalLLM,
        evidence_root: Path,
        runtime_root: Path | None = None,
    ):
        self.db = db
        self.retriever = retriever
        self.llm = llm
        self.evidence_root = Path(evidence_root)
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self.assets = PDFAssetStore(
            runtime_root or (self.evidence_root.parent / "_runtime")
        )

    def _generate(self, prompt: str, *, max_new_tokens: int, profile: str) -> str:
        """Use an explicit model tier while retaining old test/runtime adapters."""

        try:
            return self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                profile=profile,
            )
        except TypeError as exc:
            if "profile" not in str(exc):
                raise
            return self.llm.generate(prompt, max_new_tokens=max_new_tokens)

    @staticmethod
    def _batch_prompt(instruction: str, batch: list[Evidence]) -> str:
        evidence = "\n\n".join(
            f"{x.citation} 书名：{x.title}；第{x.page}页\n{x.text}"
            for x in batch
        )
        return f"""你正在处理 Phoenix 离线医学资料精确整理任务。
只能使用下面PDF证据，不得调用任何外部知识。

整理规则：
1. 只保留与用户要求直接相关的内容；相邻但无关的知识必须舍弃。
2. 每一条医学事实、数字、阈值、分级、检查技术、鉴别诊断都必须保留 [S编号] 引用。
3. 同一事实多来源重复时合并，并保留多个来源；来源冲突时并列，不自行裁决。
4. 不为了“完整”而补写证据中没有的章节；证据不足就明确写“当前资料未提供明确依据”。
5. 严格区分影像征象、诊断结论、鉴别诊断、检查前提、随访/处理建议，禁止混写。
6. 优先保留可直接用于临床阅读的具体信息，删除空泛套话。
7. 不制造新的 S 编号，不改变原始数字和单位。

用户要求：{instruction}

本批证据：
{evidence}
"""

    @staticmethod
    def _merge_prompt(title: str, instruction: str, partials: list[str]) -> str:
        joined = "\n\n===== 分批笔记 =====\n".join(partials)
        return f"""你是 Phoenix 离线医学知识工作台的最终合并整理器。
只能基于下面已经带 [S编号] 的分批笔记重组，禁止加入任何新事实。

必须做到：
- 保留来源编号，不制造新的 S 编号。
- 删除重复、空话和偏离用户要求的内容。
- 同义内容合并；不同来源冲突并列呈现，不自行判断谁正确。
- 每个关键结论都应紧跟其来源，不要把全部引用堆到文末才说明。
- 检查技术前提、影像征象、诊断倾向、鉴别、陷阱/漏诊点、报告表达分别组织。
- 没有证据支持的栏目可以省略，不能凭常识补全。
- 如果用户指定格式，以用户要求优先。

专题：{title}
用户要求：{instruction}

{joined}
"""

    @staticmethod
    def _evidence_pack(
        title: str,
        instruction: str,
        evidence: list[Evidence],
    ) -> str:
        lines = [
            f"# {title}",
            "",
            f"> 整理要求：{instruction}",
            "",
            "## PDF原文证据",
            "",
        ]
        for item in evidence:
            lines.extend([
                f"### {item.citation} {item.title} · 第{item.page}页",
                "",
                item.text,
                "",
            ])
        return "\n".join(lines)

    @staticmethod
    def _rows_to_evidence(rows) -> list[Evidence]:
        return [
            Evidence(
                chunk_id=int(row["id"]),
                source_key=str(row["source_key"]),
                title=str(row["title"]),
                path=str(row["path"]),
                page=int(row["page"]),
                text=str(row["text"]),
                score=1.0,
            )
            for row in rows
        ]

    @staticmethod
    def _query_list(title: str, instruction: str) -> list[str]:
        candidates = [
            f"{title} {instruction}".strip(),
            instruction.strip(),
            title.strip(),
        ]
        candidates.extend(
            part.strip()
            for part in _QUERY_SPLIT_RE.split(instruction)
            if len(part.strip()) >= 4
        )
        result: list[str] = []
        for item in candidates:
            item = item.strip()
            if item and item not in result:
                result.append(item)
            if len(result) >= 6:
                break
        return result

    def _retrieve_evidence(
        self,
        title: str,
        instruction: str,
        candidate_limit: int,
        *,
        progress: ProgressCallback | None = None,
    ) -> list[Evidence]:
        """Use several focused retrieval views instead of one broad query.

        A single long natural-language instruction can over-rank one book or a
        tangential phrase. Merging several conservative retrieval views improves
        coverage while still enforcing a hard final evidence limit.
        """
        queries = self._query_list(title, instruction)
        if not queries:
            return []

        per_query = max(24, min(72, int(candidate_limit)))
        merged: dict[int, Evidence] = {}
        scores: dict[int, float] = {}

        for index, query in enumerate(queries, start=1):
            if progress:
                progress(
                    index - 1,
                    len(queries),
                    f"精确检索：{index}/{len(queries)} 个主题视角",
                )
            hits = self.retriever.search_diverse(
                query,
                limit=per_query,
                use_embeddings=True,
            )
            weight = 1.0 / (1.0 + (index - 1) * 0.12)
            for hit in hits:
                score = float(hit.score) * weight
                old = scores.get(hit.chunk_id)
                if old is None or score > old:
                    merged[hit.chunk_id] = hit
                    scores[hit.chunk_id] = score

        ordered = sorted(
            merged.values(),
            key=lambda item: scores.get(item.chunk_id, 0.0),
            reverse=True,
        )

        document_count = max(1, len({item.path for item in ordered}))
        soft_cap = max(20, int(candidate_limit / min(document_count, 8)) + 8)
        counts: dict[str, int] = {}
        selected: list[Evidence] = []
        deferred: list[Evidence] = []
        for item in ordered:
            current = counts.get(item.path, 0)
            if current < soft_cap:
                selected.append(item)
                counts[item.path] = current + 1
            else:
                deferred.append(item)
            if len(selected) >= candidate_limit:
                break

        if len(selected) < candidate_limit:
            for item in deferred:
                selected.append(item)
                if len(selected) >= candidate_limit:
                    break
        return selected

    def _attach_images_inline(
        self,
        output: Path,
        evidence: list[Evidence],
        used_ids: set[int] | None = None,
        *,
        max_images: int = 120,
    ) -> int:
        """Insert source-page figures next to the first citation using them."""
        if not output.is_file():
            return 0

        selected = [
            item
            for item in evidence
            if used_ids is None or not used_ids or item.chunk_id in used_ids
        ]
        if not selected:
            return 0

        by_id = {item.chunk_id: item for item in selected}
        asset_dir = output.with_name(output.stem + "_assets")
        asset_dir.mkdir(parents=True, exist_ok=True)
        text = output.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rendered: list[str] = []
        seen_pages: set[tuple[str, int]] = set()
        image_count = 0

        for line_index, line in enumerate(lines):
            rendered.append(line)
            citation_ids = [int(x) for x in _CITATION_RE.findall(line)]
            for chunk_id in citation_ids:
                item = by_id.get(chunk_id)
                if item is None:
                    continue
                page_key = (item.path, item.page)
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
                try:
                    copied = self.assets.copy_page_assets(
                        Path(item.path),
                        item.page,
                        asset_dir,
                        prefix=f"{_safe_filename(item.title)[:32]}_",
                        ensure=True,
                    )
                except Exception:
                    copied = []
                if not copied:
                    continue

                remaining = max_images - image_count
                copied = copied[:max(0, remaining)]
                if not copied:
                    break
                rendered.extend([
                    "",
                    f"> 图像来源：{item.title} · 第{item.page}页",
                    "",
                ])
                rendered.extend(
                    markdown_images(
                        copied,
                        relative_to=output.parent,
                        label=f"{item.title} 第{item.page}页",
                    ).splitlines()
                )
                image_count += len(copied)
                if image_count >= max_images:
                    rendered.extend([
                        "",
                        f"> 相关图片超过 {max_images} 张；其余原图仍保存在本地PDF图片资料中。",
                        "",
                    ])
                    break
            if image_count >= max_images:
                rendered.extend(lines[line_index + 1:])
                break

        if image_count == 0:
            rendered.extend(["", "---", "## 相关原图", ""])
            for item in selected:
                page_key = (item.path, item.page)
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
                try:
                    copied = self.assets.copy_page_assets(
                        Path(item.path),
                        item.page,
                        asset_dir,
                        prefix=f"{_safe_filename(item.title)[:32]}_",
                        ensure=True,
                    )
                except Exception:
                    copied = []
                remaining = max_images - image_count
                copied = copied[:max(0, remaining)]
                if not copied:
                    continue
                rendered.extend([
                    f"### {item.title} · 第{item.page}页",
                    "",
                ])
                rendered.extend(
                    markdown_images(
                        copied,
                        relative_to=output.parent,
                        label=f"{item.title} 第{item.page}页",
                    ).splitlines()
                )
                image_count += len(copied)
                if image_count >= max_images:
                    break

        output.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")
        return image_count

    def _load_task_context(
        self,
        task_id: int,
    ) -> tuple[object, dict, list[Evidence]]:
        task = self.db.get_task(task_id)
        if task is None:
            raise ValueError(f"整理任务不存在: {task_id}")
        if str(task["kind"]) != "deep_organize":
            raise ValueError(f"任务 {task_id} 不是 deep_organize")

        payload = json.loads(task["payload_json"] or "{}")
        chunk_ids = [int(x) for x in payload.get("chunk_ids", [])]
        rows = self.db.fetch_chunks(chunk_ids)
        evidence = self._rows_to_evidence(rows)
        if not evidence:
            raise RuntimeError(
                "任务对应的PDF证据已经不存在，无法安全恢复。"
            )
        return task, payload, evidence

    def _pause(self, task_id: int) -> None:
        self.db.update_task(task_id, status="paused", error="")
        raise OrganizePaused(task_id)

    def _hierarchical_merge(
        self,
        title: str,
        instruction: str,
        partials: list[str],
        valid_ids: set[int],
        *,
        group_size: int = 6,
        progress: ProgressCallback | None = None,
        batch_total: int = 1,
        should_pause: PauseCallback | None = None,
        task_id: int | None = None,
    ) -> str:
        level = [text for text in partials if text.strip()]
        if not level:
            return ""
        if len(level) == 1:
            if should_pause and should_pause() and task_id is not None:
                self._pause(task_id)
            prompt_text = self._merge_prompt(title, instruction, level)
            merged = self._generate(
                prompt_text,
                max_new_tokens=_generation_budget(prompt_text, ceiling=1800),
                profile="translation",
            ).strip()
            used = {int(x) for x in _CITATION_RE.findall(merged)} & valid_ids
            return merged if used else level[0]
        merge_level = 0
        while len(level) > 1:
            merge_level += 1
            next_level: list[str] = []
            groups = [
                level[i:i + group_size]
                for i in range(0, len(level), group_size)
            ]
            for group_index, group in enumerate(groups, start=1):
                if should_pause and should_pause() and task_id is not None:
                    self._pause(task_id)
                if len(group) == 1:
                    next_level.append(group[0])
                    continue
                if progress:
                    progress(
                        batch_total,
                        batch_total,
                        f"正在精确合并：第{merge_level}层 {group_index}/{len(groups)}",
                    )
                prompt_text = self._merge_prompt(title, instruction, group)
                final_merge = len(groups) == 1
                merged = self._generate(
                    prompt_text,
                    max_new_tokens=_generation_budget(
                        prompt_text,
                        ceiling=1800 if final_merge else 1400,
                    ),
                    profile="translation" if final_merge else "fast",
                ).strip()
                used = {
                    int(x) for x in _CITATION_RE.findall(merged)
                } & valid_ids
                if not used:
                    merged = "\n\n".join(group)
                next_level.append(merged)
            level = next_level
        return level[0]

    def resume(
        self,
        task_id: int,
        *,
        progress: ProgressCallback | None = None,
        should_pause: PauseCallback | None = None,
    ) -> tuple[Path, int]:
        task, payload, _evidence = self._load_task_context(task_id)
        status = str(task["status"])
        if status not in _RESUMABLE_STATUS:
            raise RuntimeError(
                f"任务 {task_id} 当前状态为 {status}，无需恢复。"
            )
        return self.organize(
            str(payload.get("title", "医学知识专题")),
            str(payload.get("instruction", "")),
            batch_size=int(payload.get("batch_size", 12) or 12),
            task_id=int(task_id),
            progress=progress,
            should_pause=should_pause,
        )

    def organize(
        self,
        title: str,
        instruction: str,
        *,
        candidate_limit: int = 96,
        batch_size: int = 12,
        task_id: int | None = None,
        progress: ProgressCallback | None = None,
        should_pause: PauseCallback | None = None,
    ) -> tuple[Path, int]:
        title = (
            title.strip()
            or instruction.strip()[:50]
            or "医学知识专题"
        )
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("整理要求不能为空")
        candidate_limit = _env_int(
            "PHOENIX_ORGANIZE_CANDIDATE_LIMIT",
            int(candidate_limit),
            12,
            160,
        )
        batch_size = _env_int(
            "PHOENIX_ORGANIZE_BATCH_SIZE",
            int(batch_size),
            1,
            16,
        )

        if task_id is None:
            evidence = self._retrieve_evidence(
                title,
                instruction,
                candidate_limit,
                progress=progress,
            )
            if not evidence:
                output = self.evidence_root / f"{_safe_filename(title)}.md"
                output.write_text(
                    f"# {title}\n\n当前导入资料中未找到明确依据。\n",
                    encoding="utf-8",
                )
                return output, 0

            task_id = self.db.create_task(
                "deep_organize",
                {
                    "title": title,
                    "instruction": instruction,
                    "chunk_ids": [x.chunk_id for x in evidence],
                    "batch_size": batch_size,
                    "candidate_limit": candidate_limit,
                    "model_route": "fast_batches_then_quality_final",
                },
                total=(len(evidence) + batch_size - 1) // batch_size,
            )
            task = self.db.get_task(task_id)
        else:
            task, payload, evidence = self._load_task_context(task_id)
            title = str(payload.get("title") or title).strip() or title
            instruction = str(
                payload.get("instruction") or instruction
            ).strip()
            batch_size = max(
                1,
                int(payload.get("batch_size", batch_size) or batch_size),
            )

        checkpoint = (
            json.loads(task["checkpoint_json"] or "{}")
            if task else {}
        )
        partials = list(checkpoint.get("partials", []))
        next_batch = int(checkpoint.get("next_batch", len(partials)))
        total_batches = (len(evidence) + batch_size - 1) // batch_size
        next_batch = min(max(next_batch, 0), total_batches)

        self.db.update_task(
            task_id,
            status="running",
            total=total_batches,
            error="",
        )

        if not self.llm.available():
            text = self._evidence_pack(title, instruction, evidence)
            output = self.evidence_root / f"{_safe_filename(title)}.md"
            output.write_text(text, encoding="utf-8")
            image_count = self._attach_images_inline(output, evidence)
            if image_count:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"\n> 已保留 {image_count} 张引用页相关原图。\n"
                    )
            self.db.update_task(
                task_id,
                status="completed_evidence_only",
                progress=total_batches,
                checkpoint={
                    "next_batch": total_batches,
                    "partials": [],
                },
            )
            self.db.record_output(title, instruction, output)
            return output, task_id

        try:
            for batch_index in range(next_batch, total_batches):
                if should_pause and should_pause():
                    self._pause(task_id)

                batch = evidence[
                    batch_index * batch_size:
                    (batch_index + 1) * batch_size
                ]
                if progress:
                    progress(
                        batch_index,
                        total_batches,
                        f"正在精确整理第 {batch_index + 1}/{total_batches} 批证据……",
                    )
                prompt_text = self._batch_prompt(instruction, batch)
                partial = self._generate(
                    prompt_text,
                    max_new_tokens=_generation_budget(
                        prompt_text,
                        ceiling=1400,
                    ),
                    profile="fast",
                )
                valid_ids = {x.chunk_id for x in batch}
                used = {
                    int(x) for x in _CITATION_RE.findall(partial)
                } & valid_ids
                if not used:
                    partial = self._evidence_pack(
                        f"批次 {batch_index + 1}",
                        instruction,
                        batch,
                    )
                partials.append(partial)
                self.db.update_task(
                    task_id,
                    checkpoint={
                        "next_batch": batch_index + 1,
                        "partials": partials,
                    },
                    progress=batch_index + 1,
                    total=total_batches,
                )
                if progress:
                    progress(
                        batch_index + 1,
                        total_batches,
                        f"已完成证据整理 {batch_index + 1}/{total_batches} 批",
                    )
                if should_pause and should_pause():
                    self._pause(task_id)

            valid_ids = {x.chunk_id for x in evidence}
            final_text = self._hierarchical_merge(
                title,
                instruction,
                partials,
                valid_ids,
                progress=progress,
                batch_total=total_batches,
                should_pause=should_pause,
                task_id=task_id,
            )
            used_ids = {
                int(x) for x in _CITATION_RE.findall(final_text)
            } & valid_ids
            if not used_ids:
                final_text = "\n\n".join(partials)

            source_lines = ["", "---", "## 来源索引"]
            for item in evidence:
                if item.chunk_id in used_ids or not used_ids:
                    source_lines.append(
                        f"- {item.citation} {item.title}，第{item.page}页"
                    )

            output = self.evidence_root / f"{_safe_filename(title)}.md"
            output.write_text(
                f"# {title}\n\n{final_text.strip()}\n"
                + "\n".join(source_lines)
                + "\n",
                encoding="utf-8",
            )
            if progress:
                progress(
                    total_batches,
                    total_batches,
                    "正文整理完成，正在把相关原图插入对应证据附近……",
                )
            image_count = self._attach_images_inline(
                output,
                evidence,
                used_ids,
            )
            if image_count:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"\n> 本专题共保留 {image_count} 张与引用页对应的PDF原图。\n"
                    )

            self.db.update_task(
                task_id,
                status="completed",
                progress=total_batches,
            )
            self.db.record_output(title, instruction, output)
            return output, task_id
        except OrganizePaused:
            raise
        except Exception as exc:
            self.db.update_task(
                task_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
