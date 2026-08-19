from __future__ import annotations

import os

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton


_INSTALLED = False


def install(gui_module) -> None:
    """Apply lightweight GUI upgrades without duplicating the main GUI file."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow

    original_qa_tab = cls._qa_tab
    original_ask_question = cls.ask_question
    original_translation_tab = cls._translation_tab
    original_refresh_translation_models = cls.refresh_translation_models
    original_start_translation = cls.start_translation
    original_translation_done = cls._translation_done

    def _qa_tab(self):
        widget = original_qa_tab(self)
        self.deep_qa_checkbox = QCheckBox(
            "使用 Qwen3.5 深度归纳（慢；8GB显卡可能发生CPU卸载。默认关闭）"
        )
        self.deep_qa_checkbox.setChecked(False)
        layout = widget.layout()
        if layout is not None:
            layout.insertWidget(2, self.deep_qa_checkbox)
        return widget

    def ask_question(self):
        checkbox = getattr(self, "deep_qa_checkbox", None)
        if checkbox is not None and checkbox.isChecked():
            os.environ["PHOENIX_KNOWLEDGE_DEEP_QA"] = "1"
        else:
            os.environ["PHOENIX_KNOWLEDGE_DEEP_QA"] = "0"
        return original_ask_question(self)

    def _translation_tab(self):
        widget = original_translation_tab(self)
        layout = widget.layout()
        self.translation_qwen_checkbox = QCheckBox(
            "失败时启用 Qwen3.5 医学复核（慢；默认关闭，普通中文翻译使用 Marian → NLLB）"
        )
        self.translation_qwen_checkbox.setChecked(False)
        format_label = QLabel(
            "输出：TXT + Markdown + HTML；安装 python-docx 后同时输出 DOCX。"
            "PDF原图按页保存在译本 images 目录，并嵌入 Markdown / HTML / DOCX。"
        )
        format_label.setWordWrap(True)
        open_button = QPushButton("打开已有译本目录")

        def open_outputs():
            root = self.workbench.translator.output_root
            root.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

        open_button.clicked.connect(open_outputs)
        if layout is not None:
            layout.insertWidget(max(0, layout.count() - 3), self.translation_qwen_checkbox)
            layout.insertWidget(max(0, layout.count() - 3), format_label)
            layout.insertWidget(max(0, layout.count() - 3), open_button)
        return widget

    def refresh_translation_models(self):
        original_refresh_translation_models(self)
        if not hasattr(self, "translation_models_label"):
            return
        target = (
            self.translation_language.currentText()
            if hasattr(self, "translation_language")
            else "中文"
        )
        active = [x.name for x in self.workbench.translator.engine.active_backends(target)]
        available = self.workbench.translator.engine.available_backends()
        self.translation_models_label.setText(
            "默认执行链："
            + (" → ".join(active) if active else "无")
            + " | 已下载："
            + (", ".join(available) if available else "无")
        )

    def start_translation(self, retry_warning_pages: bool):
        checkbox = getattr(self, "translation_qwen_checkbox", None)
        if checkbox is not None and checkbox.isChecked():
            os.environ["PHOENIX_TRANSLATION_QWEN_REVIEW"] = "1"
        else:
            os.environ["PHOENIX_TRANSLATION_QWEN_REVIEW"] = "0"
        return original_start_translation(self, retry_warning_pages)

    def _translation_done(self, result):
        original_translation_done(self, result)
        paths = tuple(getattr(result, "output_paths", ()) or ())
        image_count = int(getattr(result, "image_count", 0) or 0)
        if paths:
            current = self.translation_result.toPlainText().rstrip()
            extra = ["", "生成格式："]
            extra.extend(f"- {path}" for path in paths)
            extra.append(f"- 原图：{image_count} 张（images目录）")
            self.translation_result.setPlainText(current + "\n" + "\n".join(extra))

    cls._qa_tab = _qa_tab
    cls.ask_question = ask_question
    cls._translation_tab = _translation_tab
    cls.refresh_translation_models = refresh_translation_models
    cls.start_translation = start_translation
    cls._translation_done = _translation_done
