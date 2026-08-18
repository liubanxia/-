from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .db import KnowledgeDB
from .llm import LocalLLM
from .retrieval import Evidence, Retriever


ProgressCallback = Callable[[int, int, str], None]
_CITATION_RE = re.compile(r"\[S(\d+)\]")
_RESUMABLE_STATUS = {"queued", "running", "failed", "paused"}


def _safe_filename(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|\r\n]+", "_", text).strip(" ._")
    return (text or "未命名专题")[:120]


class DeepOrganizer:
    def __init__(
        self,
        db: KnowledgeDB,
        retriever: Retriever,
        llm: LocalLLM,
        evidence_root: Path,
    ):
        self.db = db
        self.retriever = retriever
        self.llm = llm
        self.evidence_root = Path(evidence_root)
        self.evidence_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _batch_prompt(
        instruction: str,
        batch: list[Evidence],
    ) -> str:
        evidence = "\n\n".join(
            f"{x.citation} 书名：{x.title}；第{x.page}页\n{x.text}"
            for x in batch
        )
        return f"""你正在处理一个长期离线医学资料整理任务。
只能使用下面PDF证据，不得调用任何外部知识。
把与用户要求相关的内容整理成高密度笔记。所有事实必须保留 [S编号] 引用。
不同来源重复时合并；来源冲突时并列说明，不自行裁决。
证据不涉及的内容不要补写。

用户要求：{instruction}

本批证据：
{evidence}
"""

    @staticmethod
    def _merge_prompt(
        title: str,
        instruction: str,
        partials: list[str],
    ) -> str:
        joined = "\n\n===== 分批笔记 =====\n".join(partials)
        return f"""你是 Phoenix 离线医学知识工作台的合并整理器。
只能基于下面已经带 [S编号] 的分批笔记整理，禁止加入任何新事实。
必须保留来源编号，不得制造新的 S 编号。
去重、重组、比较；来源冲突时并列呈现，不自行裁决。
优先结构：定义/适用范围、检查技术前提、影像征象、性质判断、鉴别诊断、陷阱与漏诊点、报告表达、仍缺资料的问题。
如果用户指定格式，以用户要求优先。

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
            lines.extend(
                [
                    f"### {item.citation} {item.title} · 第{item.page}页",
                    "",
                    item.text,
                    "",
                ]
            )
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

    def _load_task_context(
        self,
        task_id: int,
    ) -> tuple[object, dict, list[Evidence]]:
        task = self.db.get_task(task_id)
        if task is None:
            raise ValueError(f"整理任务不存在: {task_id}")
        if str(task["kind"]) != "deep_organize":
            raise ValueError(
                f"任务 {task_id} 不是 deep_organize"
            )

        payload = json.loads(task["payload_json"] or "{}")
        chunk_ids = [
            int(x)
            for x in payload.get("chunk_ids", [])
        ]
        rows = self.db.fetch_chunks(chunk_ids)
        evidence = self._rows_to_evidence(rows)
        if not evidence:
            raise RuntimeError(
                "任务对应的PDF证据已经不存在，无法安全恢复。"
            )
        return task, payload, evidence

    def _hierarchical_merge(
        self,
        title: str,
        instruction: str,
        partials: list[str],
        valid_ids: set[int],
        *,
        group_size: int = 5,
        progress: ProgressCallback | None = None,
        batch_total: int = 1,
    ) -> str:
        """Merge arbitrary numbers of batch notes without one huge prompt."""

        level = [text for text in partials if text.strip()]
        if not level:
            return ""

        merge_level = 0
        while len(level) > 1:
            merge_level += 1
            next_level: list[str] = []
            groups = [
                level[i : i + group_size]
                for i in range(0, len(level), group_size)
            ]
            for group_index, group in enumerate(groups, start=1):
                if len(group) == 1:
                    next_level.append(group[0])
                    continue

                if progress:
                    progress(
                        batch_total,
                        batch_total,
                        f"正在分层合并：第{merge_level}层 "
                        f"{group_index}/{len(groups)}",
                    )

                merged = self.llm.generate(
                    self._merge_prompt(
                        title,
                        instruction,
                        group,
                    ),
                    max_new_tokens=2200,
                ).strip()
                used = {
                    int(x)
                    for x in _CITATION_RE.findall(merged)
                } & valid_ids
                if not used:
                    # Never replace grounded notes with an ungrounded merge.
                    merged = "\n\n".join(group)
                next_level.append(merged)

            level = next_level

        return level[0]

    def resume(
        self,
        task_id: int,
        *,
        progress: ProgressCallback | None = None,
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
            batch_size=int(payload.get("batch_size", 8) or 8),
            task_id=int(task_id),
            progress=progress,
        )

    def organize(
        self,
        title: str,
        instruction: str,
        *,
        candidate_limit: int = 200,
        batch_size: int = 8,
        task_id: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Path, int]:
        title = (
            title.strip()
            or instruction.strip()[:50]
            or "医学知识专题"
        )
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("整理要求不能为空")
        batch_size = max(1, int(batch_size))

        if task_id is None:
            evidence = self.retriever.search_diverse(
                instruction,
                limit=max(1, int(candidate_limit)),
                use_embeddings=True,
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
                },
                total=(len(evidence) + batch_size - 1) // batch_size,
            )
            task = self.db.get_task(task_id)
        else:
            task, payload, evidence = self._load_task_context(task_id)
            # A resumed task must use the exact original evidence set and
            # request, not a new retrieval result from a possibly changed DB.
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
            if task
            else {}
        )
        partials = list(checkpoint.get("partials", []))
        next_batch = int(
            checkpoint.get("next_batch", len(partials))
        )
        total_batches = (
            len(evidence) + batch_size - 1
        ) // batch_size
        next_batch = min(max(next_batch, 0), total_batches)

        self.db.update_task(
            task_id,
            status="running",
            total=total_batches,
            error="",
        )

        if not self.llm.available():
            text = self._evidence_pack(
                title,
                instruction,
                evidence,
            )
            output = self.evidence_root / f"{_safe_filename(title)}.md"
            output.write_text(text, encoding="utf-8")
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
            for batch_index in range(
                next_batch,
                total_batches,
            ):
                batch = evidence[
                    batch_index * batch_size :
                    (batch_index + 1) * batch_size
                ]
                partial = self.llm.generate(
                    self._batch_prompt(instruction, batch),
                    max_new_tokens=1400,
                )
                valid_ids = {x.chunk_id for x in batch}
                used = {
                    int(x)
                    for x in _CITATION_RE.findall(partial)
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

            valid_ids = {x.chunk_id for x in evidence}
            final_text = self._hierarchical_merge(
                title,
                instruction,
                partials,
                valid_ids,
                progress=progress,
                batch_total=total_batches,
            )
            used_ids = {
                int(x)
                for x in _CITATION_RE.findall(final_text)
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
            self.db.update_task(
                task_id,
                status="completed",
                progress=total_batches,
            )
            self.db.record_output(title, instruction, output)
            return output, task_id
        except Exception as exc:
            self.db.update_task(
                task_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
