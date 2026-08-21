from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from .translator import EXPORT_PDF, EXPORT_PDF_RICH

_INSTALLED = False


def _human_size(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}GB"


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

        # Replace the old product hint that advertised duplicate split PDFs as
        # the recommended output.
        for label in widget.findChildren(QLabel):
            text = label.text()
            if "同时生成一份完整PDF和按页数拆开的多册PDF" in text:
                label.setText(
                    "默认只生成一个紧凑完整PDF：复用原PDF页面对象并追加中文文字层，"
                    "不重新渲染原图。分册默认关闭；只有手动设置分册页数时才额外生成。"
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
            pdfs = [Path(path) for path in outputs if Path(path).suffix.lower() == ".pdf"]
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

            current = self.translation_result.toPlainText().rstrip()
            lines = [
                "",
                "体积检查：",
                f"- 原PDF：{_human_size(source_size)}",
                f"- 完整译本：{_human_size(complete_size)}"
                + (f"（{ratio:.2f}×）" if source_size else ""),
            ]
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
