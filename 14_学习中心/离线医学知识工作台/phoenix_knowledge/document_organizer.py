from __future__ import annotations

from pathlib import Path

from .organizer import DeepOrganizer
from .retrieval import Evidence


def _locator(item: Evidence) -> str:
    suffix = Path(item.path).suffix.lower()
    if suffix == ".pptx":
        return f"第{item.page}张幻灯片"
    if suffix == ".docx":
        return f"文档单元{item.page}"
    if suffix in {".txt", ".md"}:
        return f"文本单元{item.page}"
    return f"第{item.page}页"


class MultiDocumentOrganizer(DeepOrganizer):
    """DeepOrganizer with source-aware evidence labels for mixed documents."""

    @staticmethod
    def _batch_prompt(instruction: str, batch: list[Evidence]) -> str:
        evidence = "\n\n".join(
            f"{item.citation} 资料：{item.title}；{_locator(item)}\n{item.text}"
            for item in batch
        )
        return f"""你正在处理 Phoenix 离线医学资料精确整理任务。
只能使用下面医学资料证据，不得调用任何外部知识。

整理规则：
1. 只保留与用户要求直接相关的内容；相邻但无关的知识必须舍弃。
2. 每一条医学事实、数字、阈值、分级、检查技术、鉴别诊断都必须保留 [S编号] 引用。
3. 同一事实多来源重复时合并，并保留多个来源；来源冲突时并列，不自行裁决。
4. 不为了“完整”而补写证据中没有的章节；证据不足就明确写“当前资料未提供明确依据”。
5. 严格区分影像征象、诊断结论、鉴别诊断、检查前提、随访/处理建议，禁止混写。
6. 优先保留可直接用于临床阅读的具体信息，删除空泛套话。
7. 不制造新的 S 编号，不改变原始数字和单位。
8. PDF页码、PPTX幻灯片编号、DOCX/文本单元编号都属于来源定位信息，必须保留。

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

    def _attach_images_inline(self, *args, **kwargs) -> int:
        count = super()._attach_images_inline(*args, **kwargs)
        if args:
            output = Path(args[0])
        else:
            output = Path(kwargs.get("output", ""))
        if output.is_file():
            text = output.read_text(encoding="utf-8", errors="replace")
            text = text.replace(
                "其余原图仍保存在本地PDF图片资料中。",
                "其余原图仍保存在本地资料图片缓存中。",
            )
            output.write_text(text, encoding="utf-8")
        return count
