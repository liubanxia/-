from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QLabel

from .translation_layout_compact import LAYOUT_SOURCE_TRANSLATED
from .translation_pdf import (
    LAYOUT_ORIGINAL_BILINGUAL,
    LAYOUT_TEXT_BILINGUAL,
    LAYOUT_TRANSLATED_ONLY,
)
from .translator import EXPORT_PDF, EXPORT_PDF_RICH

_INSTALLED = False


def _human_size(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}GB"


def _release_ratio_target(layout: str) -> float:
    if str(layout) in {
        LAYOUT_ORIGINAL_BILINGUAL,
        LAYOUT_TEXT_BILINGUAL,
    }:
        return 1.50
    return 1.30


def _integrity_summary(complete: Path) -> str | None:
    report_path = Path(complete).parent / "PDF完整性报告.json"
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return "- 完整性验收：报告无法读取（成品不会因此被标记为PASS）"
    if not bool(payload.get("passed", False)):
        return "- 完整性验收：FAIL"
    pdf = payload.get("pdf") or {}
    min_coverage = pdf.get("translation_coverage_min")
    text = "- 完整性验收：PASS（可打开、页数、文字层、原图资源）"
    if min_coverage is not None:
        try:
            text += f"；最低译文覆盖率 {float(min_coverage):.0%}"
        except Exception:
            pass
    return text


def install(gui_module) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow
    original_translation_tab = cls._translation_tab
    original_translation_done = cls._translation_done

    def _translation_tab(self):
        widget = original_translation_tab(self)

        if hasattr(self, "translation_layout_combo"):
            combo = self.translation_layout_combo
            if combo.findData(LAYOUT_SOURCE_TRANSLATED) < 0:
                combo.insertItem(
                    0,
                    "原版图文中文译本（推荐，体积接近原PDF）",
                    LAYOUT_SOURCE_TRANSLATED,
                )
            legacy_index = combo.findData(LAYOUT_ORIGINAL_BILINGUAL)
            if legacy_index >= 0:
                combo.setItemText(
                    legacy_index,
                    "上下双语版：原PDF页 + 中文译文（页面更长）",
                )
            compact_index = combo.findData(LAYOUT_SOURCE_TRANSLATED)
            if compact_index >= 0:
                combo.setCurrentIndex(compact_index)
            combo.setToolTip(
                "推荐模式直接复用原PDF图片、矢量图和页面尺寸，只替换可识别文字层；"
                "扫描页或极复杂页面才使用紧凑页尾译文区。"
            )

        if hasattr(self, "translation_export_combo"):
            combo = self.translation_export_combo
            for index in range(combo.count()):
                value = combo.itemData(index)
                if value == EXPORT_PDF:
                    combo.setItemText(index, "PDF整书（推荐，省空间）")
                elif value == EXPORT_PDF_RICH:
                    combo.setItemText(
                        index,
                        "PDF + DOCX + Markdown + TXT（额外占空间）",
                    )

        if hasattr(self, "translation_part_pages"):
            spin = self.translation_part_pages
            spin.setRange(0, 200)
            spin.setSpecialValueText("不生成分册")
            spin.setValue(0)
            spin.setToolTip(
                "0=只生成一个完整PDF；只有需要拆册时才填写页数。"
                "生成分册会额外占用接近一整本PDF的磁盘空间。"
            )

        for label in widget.findChildren(QLabel):
            text = label.text()
            if (
                "同时生成一份完整PDF和按页数拆开的多册PDF" in text
                or "复用原PDF页面对象并追加中文文字层" in text
            ):
                label.setText(
                    "推荐“原版图文中文译本”：直接保留原PDF图片、矢量图、表格和页面尺寸，"
                    "删除原文字层后在相同文字区域写入中文；不复制整页、不重新渲染原图。"
                    "默认不生成分册。发布体积目标：中文译本通常≤1.30×；保留原页双语版≤1.50×。"
                )
                label.setWordWrap(True)

        return widget

    def _translation_done(self, result):
        original_translation_done(self, result)
        if bool(getattr(result, "paused", False)):
            return

        try:
            source = Path(result.source_path)
            outputs = tuple(getattr(result, "output_paths", ()) or ())
            pdfs = [
                Path(path)
                for path in outputs
                if Path(path).suffix.lower() == ".pdf"
            ]
            if not pdfs or not source.is_file():
                return

            complete = pdfs[0]
            if not complete.is_file():
                return
            source_size = int(source.stat().st_size)
            complete_size = int(complete.stat().st_size)
            ratio = complete_size / source_size if source_size else 0.0
            extra_parts = sum(
                int(path.stat().st_size)
                for path in pdfs[1:]
                if path.is_file()
            )
            target_ratio = _release_ratio_target(
                str(getattr(result, "output_layout", LAYOUT_SOURCE_TRANSLATED))
            )

            current = self.translation_result.toPlainText().rstrip()
            lines = [
                "",
                "成品验收：",
                f"- 原PDF：{_human_size(source_size)}",
                f"- 完整译本：{_human_size(complete_size)}"
                + (f"（{ratio:.2f}×）" if source_size else ""),
            ]
            if source_size and ratio <= target_ratio:
                lines.append(
                    f"- 发布体积目标：PASS（≤{target_ratio:.2f}×）"
                )
            elif source_size:
                lines.append(
                    f"- 发布体积目标：FAIL（目标≤{target_ratio:.2f}×，"
                    "请检查特殊字体/新增资源/复杂页面）"
                )

            integrity = _integrity_summary(complete)
            if integrity:
                lines.append(integrity)
            else:
                lines.append(
                    "- 完整性验收：未找到报告；不应把该PDF视为稳定发布成品"
                )

            if extra_parts:
                lines.append(
                    f"- 分册额外占用：{_human_size(extra_parts)}"
                )
            else:
                lines.append("- 分册：未生成，不重复占用一整本空间")
            self.translation_result.setPlainText(
                current + "\n" + "\n".join(lines)
            )
        except Exception:
            pass

    cls._translation_tab = _translation_tab
    cls._translation_done = _translation_done
