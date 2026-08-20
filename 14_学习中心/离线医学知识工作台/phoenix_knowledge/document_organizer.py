from __future__ import annotations

import json
from pathlib import Path

from .organizer import DeepOrganizer
from .retrieval import Evidence


_SCHOLARLY_FULLTEXT = {".html", ".htm", ".xml", ".nxml", ".jats"}
_SCHOLARLY_METADATA = {".nbib", ".ris", ".bib", ".bibtex", ".json", ".csljson"}
_CNKI_PAPER = {".caj", ".nh", ".hn", ".kdh", ".teb", ".c8"}


def _locator(item: Evidence) -> str:
    suffix = Path(item.path).suffix.lower()
    if suffix in {".ppt", ".pptx"}:
        return f"第{item.page}张幻灯片"
    if suffix == ".docx":
        return f"文档单元{item.page}"
    if suffix in {".txt", ".md"}:
        return f"文本单元{item.page}"
    if suffix in _SCHOLARLY_FULLTEXT:
        return f"论文单元{item.page}"
    if suffix in _SCHOLARLY_METADATA:
        return f"文献记录{item.page}"
    if suffix in _CNKI_PAPER:
        return f"第{item.page}页"
    return f"第{item.page}页"


class MultiDocumentOrganizer(DeepOrganizer):
    """DeepOrganizer with source-aware evidence labels for mixed documents."""

    @staticmethod
    def _batch_prompt(
        instruction: str,
        batch: list[Evidence],
    ) -> str:
        evidence = "\n\n".join(
            f"{item.citation} 资料：{item.title}；{_locator(item)}\n{item.text}"
            for item in batch
        )
        return f"""你正在处理 Phoenix 离线医学资料精确整理任务。
只能使用下面医学资料证据，不得调用任何外部知识。

整理规则：
1. 只保留与用户要求直接相关的内容；相邻但无关的知识必须舍弃。
2. 每一条医学事实、数字、阈值、分级、检查技术、鉴别诊断、论文研究结果都必须保留 [S编号] 引用。
3. 同一事实多来源重复时合并，并保留多个来源；来源冲突时并列，不自行裁决。
4. 不为了“完整”而补写证据中没有的章节；证据不足就明确写“当前资料未提供明确依据”。
5. 严格区分影像征象、诊断结论、鉴别诊断、检查前提、研究设计、样本量、统计结果、局限性与随访/处理建议。
6. 论文中的 DOI/PMID/PMCID、样本量、敏感度、特异度、AUC、置信区间和 P 值属于高价值证据，存在时应准确保留。
7. 不制造新的 S 编号，不改变原始数字和单位。
8. PDF/CAJ类论文页码、PPT/PPTX幻灯片编号、DOCX/文本单元、论文单元和题录记录号都属于来源定位信息，必须保留。
9. 题录格式（NBIB/RIS/BibTeX/CSL-JSON）只代表元数据/摘要证据，不能把未提供的全文内容当成已阅读全文。

用户要求：{instruction}

本批证据：
{evidence}
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
            "## 医学资料原文证据",
            "",
        ]
        for item in evidence:
            lines.extend(
                [
                    f"### {item.citation} {item.title} · {_locator(item)}",
                    "",
                    item.text,
                    "",
                ]
            )
        return "\n".join(lines)

    def _task_evidence(
        self,
        task_id: int,
    ) -> list[Evidence]:
        task = self.db.get_task(task_id)
        if task is None:
            return []
        try:
            payload = json.loads(
                task["payload_json"] or "{}"
            )
            chunk_ids = [
                int(value)
                for value in payload.get(
                    "chunk_ids",
                    [],
                )
            ]
        except Exception:
            return []
        rows = self.db.fetch_chunks(chunk_ids)
        return self._rows_to_evidence(rows)

    def _normalize_output_locators(
        self,
        output: Path,
        task_id: int,
    ) -> None:
        if not output.is_file() or not task_id:
            return
        evidence = self._task_evidence(task_id)
        if not evidence:
            return
        text = output.read_text(
            encoding="utf-8",
            errors="replace",
        )
        for item in evidence:
            correct = _locator(item)
            legacy = f"第{item.page}页"
            text = text.replace(
                f"- {item.citation} {item.title}，{legacy}",
                f"- {item.citation} {item.title}，{correct}",
            )
            text = text.replace(
                f"图像来源：{item.title} · {legacy}",
                f"图像来源：{item.title} · {correct}",
            )
            text = text.replace(
                f"### {item.title} · {legacy}",
                f"### {item.title} · {correct}",
            )
            text = text.replace(
                f"![{item.title} {legacy} 图",
                f"![{item.title} {correct} 图",
            )
        output.write_text(
            text,
            encoding="utf-8",
        )

    def _attach_images_inline(
        self,
        *args,
        **kwargs,
    ) -> int:
        count = super()._attach_images_inline(
            *args,
            **kwargs,
        )
        if args:
            output = Path(args[0])
        else:
            output = Path(
                kwargs.get("output", "")
            )
        if output.is_file():
            text = output.read_text(
                encoding="utf-8",
                errors="replace",
            )
            text = text.replace(
                "其余原图仍保存在本地PDF图片资料中。",
                "其余原图仍保存在本地资料图片缓存中。",
            )
            output.write_text(
                text,
                encoding="utf-8",
            )
        return count

    def organize(self, *args, **kwargs):
        output, task_id = super().organize(
            *args,
            **kwargs,
        )
        if task_id:
            self._normalize_output_locators(
                Path(output),
                int(task_id),
            )
        return output, task_id
