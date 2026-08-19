from __future__ import annotations

import os
import time

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton


_INSTALLED = False


def install(gui_module) -> None:
    """Apply product/responsiveness upgrades without duplicating the main GUI."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow
    ask_worker_cls = gui_module.AskWorker

    original_init = cls.__init__
    original_qa_tab = cls._qa_tab
    original_ask_question = cls.ask_question
    original_translation_tab = cls._translation_tab
    original_refresh_translation_models = cls.refresh_translation_models
    original_start_translation = cls.start_translation
    original_translation_done = cls._translation_done

    def _ask_worker_run(self):
        """Two-stage Q&A: lexical evidence first, semantic/deep result second."""
        try:
            started = time.perf_counter()
            quick = self.workbench.ask(
                self.query,
                use_embeddings=False,
                deep=False,
            )
            quick_elapsed = time.perf_counter() - started
            self.completed.emit(
                f"【即时检索 {quick_elapsed:.2f}s】\n"
                f"{quick.text}\n\n"
                "—— 正在后台做语义补全；如启用智能归纳，将在语义检索后继续生成 ——"
            )

            full_started = time.perf_counter()
            full = self.workbench.ask(self.query)
            full_elapsed = time.perf_counter() - full_started
            model = self.workbench.llm.active_model_name()
            self.completed.emit(
                f"【完成 | 模式={full.mode} | 第二阶段 {full_elapsed:.2f}s"
                f" | 生成模型={model}】\n\n{full.text}"
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    ask_worker_cls.run = _ask_worker_run

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            from .licensing import LicenseManager

            manager = LicenseManager(self.workbench.paths.project_root)
            status = manager.status()
            if status.product_mode and status.valid:
                edition = status.edition or "Professional"
                customer = status.customer or "正式授权"
                self.setWindowTitle(f"Phoenix 医学知识工作台 · {edition}")
                license_label = QLabel(
                    f"已激活 · {edition} · {customer} · 授权号 {status.license_id}"
                )
                self.statusBar().addPermanentWidget(license_label)
            else:
                self.setWindowTitle("Phoenix 医学知识工作台 · Development")
                license_label = QLabel("开发模式 · 正式发布时启用离线激活")
                self.statusBar().addPermanentWidget(license_label)
        except Exception:
            pass

    def _qa_tab(self):
        widget = original_qa_tab(self)
        layout = widget.layout()

        self.deep_qa_checkbox = QCheckBox(
            "智能归纳（优先 Qwen3.5-2B；未下载时回退4B。默认关闭）"
        )
        self.deep_qa_checkbox.setChecked(False)
        self.force_4b_checkbox = QCheckBox(
            "强制 Qwen3.5-4B 深度质量模式（最慢；8GB显卡可能CPU卸载）"
        )
        self.force_4b_checkbox.setChecked(False)
        self.force_4b_checkbox.setEnabled(False)
        self.deep_qa_checkbox.toggled.connect(self.force_4b_checkbox.setEnabled)

        fast_name = self.workbench.llm.active_model_name("fast")
        deep_name = self.workbench.llm.active_model_name("deep")
        mode_label = QLabel(
            "响应链：SQLite即时证据 → Embedding语义补全 → 可选智能归纳。"
            f" 快速生成={fast_name}；深度生成={deep_name}。"
        )
        mode_label.setWordWrap(True)

        if layout is not None:
            layout.insertWidget(2, mode_label)
            layout.insertWidget(3, self.deep_qa_checkbox)
            layout.insertWidget(4, self.force_4b_checkbox)
        return widget

    def ask_question(self):
        checkbox = getattr(self, "deep_qa_checkbox", None)
        force_4b = getattr(self, "force_4b_checkbox", None)
        intelligent = checkbox is not None and checkbox.isChecked()
        deep4b = intelligent and force_4b is not None and force_4b.isChecked()

        os.environ["PHOENIX_KNOWLEDGE_DEEP_QA"] = "1" if intelligent else "0"
        os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = "deep" if deep4b else "fast"
        return original_ask_question(self)

    def _translation_tab(self):
        widget = original_translation_tab(self)
        layout = widget.layout()
        self.translation_qwen_checkbox = QCheckBox(
            "失败时启用 Qwen 医学复核（优先2B；默认关闭，普通中文翻译使用 Marian → NLLB）"
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
        qwen_review = checkbox is not None and checkbox.isChecked()
        os.environ["PHOENIX_TRANSLATION_QWEN_REVIEW"] = "1" if qwen_review else "0"
        os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = "fast"

        # Translation should not compete with stale embedding/Qwen weights for
        # VRAM. Dedicated Marian/NLLB models are loaded only after this release.
        self.workbench.retriever.embeddings.unload_model()
        self.workbench.llm.unload()
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

    cls.__init__ = _init
    cls._qa_tab = _qa_tab
    cls.ask_question = ask_question
    cls._translation_tab = _translation_tab
    cls.refresh_translation_models = refresh_translation_models
    cls.start_translation = start_translation
    cls._translation_done = _translation_done
