from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from .organizer import OrganizePaused
from .translation_pdf import (
    LAYOUT_ORIGINAL_BILINGUAL,
    LAYOUT_TEXT_BILINGUAL,
    LAYOUT_TRANSLATED_ONLY,
)
from .translator import (
    EXPORT_PDF,
    EXPORT_PDF_RICH,
    EXPORT_RICH,
    EXPORT_TXT,
)


_INSTALLED = False


class _OrganizeWorkerV2(QThread):
    progress = Signal(int, int, str)
    completed = Signal(str)
    paused = Signal(int)
    failed = Signal(str)

    def __init__(
        self,
        workbench,
        *,
        title: str = "",
        instruction: str = "",
        task_id: int | None = None,
    ):
        super().__init__()
        self.workbench = workbench
        self.title = title
        self.instruction = instruction
        self.task_id = task_id
        self._pause_requested = threading.Event()

    def request_pause(self) -> None:
        self._pause_requested.set()

    def run(self):
        try:
            callback = lambda done, total, msg: self.progress.emit(
                int(done), int(total), str(msg)
            )
            if self.task_id is None:
                output, _task_id = self.workbench.organize(
                    self.title,
                    self.instruction,
                    progress=callback,
                    should_pause=self._pause_requested.is_set,
                )
            else:
                output, _task_id = self.workbench.resume_task(
                    self.task_id,
                    progress=callback,
                    should_pause=self._pause_requested.is_set,
                )
            self.completed.emit(str(output))
        except OrganizePaused as exc:
            self.paused.emit(int(exc.task_id))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _TranslationWorkerV2(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        workbench,
        pdf_path: str,
        start_page: int,
        target_language: str,
        *,
        retry_warning_pages: bool,
        smart_level: str,
        output_layout: str,
        export_format: str,
        part_pages: int,
    ):
        super().__init__()
        self.workbench = workbench
        self.pdf_path = pdf_path
        self.start_page = int(start_page)
        self.target_language = target_language
        self.retry_warning_pages = bool(retry_warning_pages)
        self.smart_level = smart_level
        self.output_layout = output_layout
        self.export_format = export_format
        self.part_pages = int(part_pages)
        self._pause_requested = threading.Event()

    def request_pause(self) -> None:
        self._pause_requested.set()

    def run(self):
        try:
            result = self.workbench.translate_book(
                Path(self.pdf_path),
                start_page=self.start_page,
                target_language=self.target_language,
                retry_warning_pages=self.retry_warning_pages,
                smart_level=self.smart_level,
                output_layout=self.output_layout,
                export_format=self.export_format,
                part_pages=self.part_pages,
                should_pause=self._pause_requested.is_set,
                progress=lambda done, total, msg: self.progress.emit(
                    int(done), int(total), str(msg)
                ),
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def install(gui_module) -> None:
    """Apply product, readability and long-task controls to the workbench GUI."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow
    ask_worker_cls = gui_module.AskWorker

    original_init = cls.__init__
    original_qa_tab = cls._qa_tab
    original_ask_question = cls.ask_question
    original_organize_tab = cls._organize_tab
    original_organize_done = cls._organize_done
    original_organize_failed = cls._organize_failed
    original_translation_tab = cls._translation_tab
    original_translation_failed = cls._translation_failed
    original_print_translation = cls._print_translation

    def _status_text(self) -> str:
        status = self.workbench.status()
        smart1 = "READY" if self.workbench.llm.available("fast") else "未就绪"
        smart2 = "READY" if self.workbench.llm.available("deep") else "未就绪"
        semantic = "READY" if status.get("embedding_available") else "未就绪"
        return (
            f"资料 {status['documents']} 本 | 知识块 {status['chunks']} | "
            f"语义检索={semantic} | 智能1={smart1} | 智能2={smart2}"
        )

    cls._status_text = _status_text

    def _ask_worker_run(self):
        """Two-stage Q&A: immediate evidence first, semantic/intelligent result second."""
        try:
            started = time.perf_counter()
            quick = self.workbench.ask(
                self.query,
                use_embeddings=False,
                deep=False,
            )
            quick_elapsed = time.perf_counter() - started
            self.completed.emit(
                f"【即时证据 {quick_elapsed:.2f}s】\n"
                f"{quick.text}\n\n"
                "—— 正在后台做语义补全；如选择智能1/智能2，将继续进行证据归纳 ——"
            )

            full_started = time.perf_counter()
            full = self.workbench.ask(self.query)
            full_elapsed = time.perf_counter() - full_started
            deep_enabled = os.environ.get(
                "PHOENIX_KNOWLEDGE_DEEP_QA", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            profile = os.environ.get(
                "PHOENIX_KNOWLEDGE_LLM_PROFILE", "fast"
            ).strip().lower()
            if not deep_enabled:
                label = "快速证据"
            elif profile in {"deep", "4b", "deep4b", "quality", "max"}:
                label = "智能2"
            else:
                label = "智能1"
            self.completed.emit(
                f"【完成 | {label} | 第二阶段 {full_elapsed:.2f}s】\n\n"
                f"{full.text}"
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

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("回答方式："))
        self.qa_mode_combo = QComboBox()
        self.qa_mode_combo.addItem("快速证据", "evidence")
        self.qa_mode_combo.addItem("智能1", "smart1")
        self.qa_mode_combo.addItem("智能2", "smart2")
        self.qa_mode_combo.setCurrentIndex(0)
        mode_row.addWidget(self.qa_mode_combo)
        mode_row.addStretch(1)

        mode_label = QLabel(
            "响应链：即时PDF证据 → 语义检索 → 可选智能1/智能2。"
            "界面不显示底层模型名称。"
        )
        mode_label.setWordWrap(True)

        if layout is not None:
            layout.insertLayout(2, mode_row)
            layout.insertWidget(3, mode_label)
        return widget

    def ask_question(self):
        mode = (
            self.qa_mode_combo.currentData()
            if hasattr(self, "qa_mode_combo")
            else "evidence"
        )
        intelligent = mode in {"smart1", "smart2"}
        os.environ["PHOENIX_KNOWLEDGE_DEEP_QA"] = (
            "1" if intelligent else "0"
        )
        os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = (
            "deep" if mode == "smart2" else "fast"
        )
        return original_ask_question(self)

    def _organize_tab(self):
        widget = original_organize_tab(self)
        layout = widget.layout()

        row = QHBoxLayout()
        row.addWidget(QLabel("整理方式："))
        self.organize_smart_combo = QComboBox()
        self.organize_smart_combo.addItem("智能1", "smart1")
        self.organize_smart_combo.addItem("智能2", "smart2")
        self.organize_smart_combo.setCurrentIndex(0)
        row.addWidget(self.organize_smart_combo)
        self.pause_organize_button = QPushButton("暂停整理")
        self.pause_organize_button.setEnabled(False)
        self.pause_organize_button.clicked.connect(self.pause_organize)
        row.addWidget(self.pause_organize_button)
        row.addStretch(1)

        note = QLabel(
            "多资料整理会从多个检索视角交叉取证；暂停后保留已完成批次，"
            "点击“继续未完成任务”即可续做。相关PDF原图会插入到引用内容附近。"
        )
        note.setWordWrap(True)
        if layout is not None:
            layout.insertLayout(1, row)
            layout.insertWidget(2, note)
        return widget

    def _connect_organize_worker(self):
        self.worker.progress.connect(self._organize_progress)
        self.worker.completed.connect(self._organize_done)
        self.worker.failed.connect(self._organize_failed)
        if hasattr(self.worker, "paused"):
            self.worker.paused.connect(self._organize_paused)
        if hasattr(self, "pause_organize_button"):
            self.pause_organize_button.setEnabled(True)
        self.worker.start()

    def start_organize(self):
        if self._busy():
            return
        title = self.topic_title.text().strip()
        instruction = self.organize_edit.toPlainText().strip()
        if not instruction:
            return
        mode = (
            self.organize_smart_combo.currentData()
            if hasattr(self, "organize_smart_combo")
            else "smart1"
        )
        os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = (
            "deep" if mode == "smart2" else "fast"
        )
        self.organize_progress.setValue(0)
        self.organize_result.setPlainText(
            "正在从全部PDF进行多视角精确检索与图文整理……"
        )
        self.worker = _OrganizeWorkerV2(
            self.workbench,
            title=title,
            instruction=instruction,
        )
        self._connect_organize_worker()

    def resume_organize(self):
        if self._busy():
            return
        task = self.workbench.latest_resumable_task()
        if task is None:
            self.refresh_resume_state()
            return
        import json

        payload = json.loads(task["payload_json"] or "{}")
        self.topic_title.setText(str(payload.get("title", "")))
        self.organize_edit.setPlainText(str(payload.get("instruction", "")))
        total = max(int(task["total"]), 1)
        done = int(task["progress"])
        self.organize_progress.setValue(int(done / total * 100))
        self.organize_status.setText(
            f"正在从checkpoint继续任务 #{int(task['id'])}……"
        )
        self.organize_result.setPlainText("正在恢复多资料整理任务……")
        mode = (
            self.organize_smart_combo.currentData()
            if hasattr(self, "organize_smart_combo")
            else "smart1"
        )
        os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = (
            "deep" if mode == "smart2" else "fast"
        )
        self.worker = _OrganizeWorkerV2(
            self.workbench,
            task_id=int(task["id"]),
        )
        self._connect_organize_worker()

    def pause_organize(self):
        worker = getattr(self, "worker", None)
        if isinstance(worker, _OrganizeWorkerV2) and worker.isRunning():
            worker.request_pause()
            self.organize_status.setText(
                "已请求暂停：完成当前批次/当前生成后停止，checkpoint不会丢失。"
            )
            self.pause_organize_button.setEnabled(False)

    def _organize_paused(self, task_id: int):
        if hasattr(self, "pause_organize_button"):
            self.pause_organize_button.setEnabled(False)
        self.organize_status.setText(
            f"任务 #{int(task_id)} 已暂停；点击“继续未完成任务”恢复。"
        )
        self.refresh_resume_state()

    def _organize_done(self, output: str):
        if hasattr(self, "pause_organize_button"):
            self.pause_organize_button.setEnabled(False)
        return original_organize_done(self, output)

    def _organize_failed(self, error: str):
        if hasattr(self, "pause_organize_button"):
            self.pause_organize_button.setEnabled(False)
        return original_organize_failed(self, error)

    def _translation_tab(self):
        widget = original_translation_tab(self)
        layout = widget.layout()

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("翻译方式："))
        self.translation_smart_combo = QComboBox()
        self.translation_smart_combo.addItem(
            "医学精译（质量模型，低推理）",
            "smart2",
        )
        quality_row.addWidget(self.translation_smart_combo)
        quality_row.addStretch(1)

        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("译本版式："))
        self.translation_layout_combo = QComboBox()
        self.translation_layout_combo.addItem(
            "原PDF页在上 / 中文译文在下（推荐）",
            LAYOUT_ORIGINAL_BILINGUAL,
        )
        self.translation_layout_combo.addItem(
            "英文文本在上 / 中文译文在下",
            LAYOUT_TEXT_BILINGUAL,
        )
        self.translation_layout_combo.addItem(
            "仅中文译文",
            LAYOUT_TRANSLATED_ONLY,
        )
        layout_row.addWidget(self.translation_layout_combo, 1)

        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("输出："))
        self.translation_export_combo = QComboBox()
        self.translation_export_combo.addItem(
            "PDF整书 + PDF分册（推荐）",
            EXPORT_PDF,
        )
        self.translation_export_combo.addItem(
            "PDF + DOCX + Markdown + TXT",
            EXPORT_PDF_RICH,
        )
        self.translation_export_combo.addItem(
            "DOCX + Markdown + TXT",
            EXPORT_RICH,
        )
        self.translation_export_combo.addItem("仅TXT", EXPORT_TXT)
        export_row.addWidget(self.translation_export_combo, 1)
        export_row.addWidget(QLabel("每册原文页数："))
        self.translation_part_pages = QSpinBox()
        self.translation_part_pages.setRange(10, 200)
        self.translation_part_pages.setValue(50)
        export_row.addWidget(self.translation_part_pages)

        action_row = QHBoxLayout()
        self.pause_translation_button = QPushButton("暂停翻译")
        self.pause_translation_button.setEnabled(False)
        self.pause_translation_button.clicked.connect(self.pause_translation)
        open_button = QPushButton("打开译本目录")

        def open_outputs():
            root = self.workbench.translator.output_root
            root.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

        open_button.clicked.connect(open_outputs)
        action_row.addWidget(self.pause_translation_button)
        action_row.addWidget(open_button)
        action_row.addStretch(1)

        format_label = QLabel(
            "默认成品会保留原PDF页面及原图，中文译文紧接在原页下方；"
            "同时生成一份完整PDF和按页数拆开的多册PDF。暂停后已完成页不会重翻。"
        )
        format_label.setWordWrap(True)

        if layout is not None:
            layout.insertLayout(3, quality_row)
            layout.insertLayout(4, layout_row)
            layout.insertLayout(5, export_row)
            layout.insertLayout(6, action_row)
            layout.insertWidget(7, format_label)
        return widget

    def refresh_translation_models(self):
        if not hasattr(self, "translation_models_label"):
            return
        quality = (
            "可用"
            if self.workbench.llm.available("translation")
            else "未就绪"
        )
        engine = self.workbench.translator.engine
        preview = (
            "可用"
            if engine.marian.available() or engine.nllb.available()
            else "未就绪"
        )
        self.translation_models_label.setText(
            f"医学精译={quality}（仅质量模型） | "
            f"普通资料快速预览={preview}"
        )

    def start_translation(self, retry_warning_pages: bool):
        if self._busy():
            return
        path = self.translation_path.text().strip()
        if not path:
            self.choose_translation_pdf()
            path = self.translation_path.text().strip()
        if not path:
            return

        self.refresh_translation_models()
        self.translation_progress.setValue(0)
        self.translation_result.setPlainText(
            "整本医学精译任务正在运行；每完成一页立即保存……"
        )
        smart_level = (
            self.translation_smart_combo.currentData()
            if hasattr(self, "translation_smart_combo")
            else "smart2"
        )
        output_layout = (
            self.translation_layout_combo.currentData()
            if hasattr(self, "translation_layout_combo")
            else LAYOUT_ORIGINAL_BILINGUAL
        )
        export_format = (
            self.translation_export_combo.currentData()
            if hasattr(self, "translation_export_combo")
            else EXPORT_PDF
        )
        part_pages = (
            self.translation_part_pages.value()
            if hasattr(self, "translation_part_pages")
            else 50
        )
        os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = (
            "translation" if smart_level == "smart2" else "fast"
        )

        # Release unrelated resident weights before the translation engine loads
        # its selected intelligent/fallback backend.
        self.workbench.retriever.embeddings.unload_model()
        self.workbench.llm.unload()

        self.worker = _TranslationWorkerV2(
            self.workbench,
            path,
            self.translation_start_page.value(),
            self.translation_language.currentText(),
            retry_warning_pages=retry_warning_pages,
            smart_level=str(smart_level),
            output_layout=str(output_layout),
            export_format=str(export_format),
            part_pages=int(part_pages),
        )
        self.worker.progress.connect(self._translation_progress)
        self.worker.completed.connect(self._translation_done)
        self.worker.failed.connect(self._translation_failed)
        self.pause_translation_button.setEnabled(True)
        self.worker.start()

    def pause_translation(self):
        worker = getattr(self, "worker", None)
        if isinstance(worker, _TranslationWorkerV2) and worker.isRunning():
            worker.request_pause()
            self.translation_status.setText(
                "已请求暂停：完成当前页后停止，已完成页不会丢失。"
            )
            self.pause_translation_button.setEnabled(False)

    def _translation_done(self, result):
        if hasattr(self, "pause_translation_button"):
            self.pause_translation_button.setEnabled(False)

        if bool(getattr(result, "paused", False)):
            self.translation_status.setText(
                "翻译已暂停；已完成页已写入checkpoint。再次点击“开始/继续整本翻译”即可续翻。"
            )
            self.translation_result.setPlainText(
                f"已暂停\n已处理页：{result.pages_done}\n"
                f"已续用页：{result.resumed_pages}\n"
                f"待复核页：{result.warning_pages}\n\n"
                "已完成内容保留在当前译本任务目录，不会从头重来。"
            )
            self.refresh_translation_models()
            return

        paths = tuple(getattr(result, "output_paths", ()) or ())
        primary = Path(paths[0]) if paths else Path(result.output_path)
        self.last_translation_path = primary
        self.translation_progress.setValue(100)
        smart_label = (
            "医学精译"
            if str(getattr(result, "smart_level", "smart2")) == "smart2"
            else "普通资料快速预览"
        )
        self.translation_status.setText(
            f"整本翻译完成 | {smart_label} | 待复核页={result.warning_pages}"
        )
        lines = [
            "翻译完成。",
            f"原文件：{result.source_path}",
            f"范围：第 {result.start_page} 页至第 {result.total_pages} 页",
            f"续翻跳过页：{result.resumed_pages}",
            f"待复核页：{result.warning_pages}",
            "",
            "成品文件：",
        ]
        lines.extend(f"- {path}" for path in paths)
        if not paths:
            lines.append(f"- {result.output_path}")
        if int(getattr(result, "image_count", 0) or 0):
            lines.append(
                f"- 可编辑格式附带原图：{int(result.image_count)} 张"
            )
        self.translation_result.setPlainText("\n".join(lines))
        self.refresh_translation_models()

    def _translation_failed(self, error: str):
        if hasattr(self, "pause_translation_button"):
            self.pause_translation_button.setEnabled(False)
        return original_translation_failed(self, error)

    def _print_translation(self, preview: bool):
        path = getattr(self, "last_translation_path", None)
        if path is not None:
            path = Path(path)
            if path.is_file() and path.suffix.lower() == ".pdf":
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
                self.translation_status.setText(
                    "已用系统PDF阅读器打开译本，可直接打印或另存。"
                )
                return
        return original_print_translation(self, preview)

    cls.__init__ = _init
    cls._qa_tab = _qa_tab
    cls.ask_question = ask_question
    cls._organize_tab = _organize_tab
    cls._connect_organize_worker = _connect_organize_worker
    cls.start_organize = start_organize
    cls.resume_organize = resume_organize
    cls.pause_organize = pause_organize
    cls._organize_paused = _organize_paused
    cls._organize_done = _organize_done
    cls._organize_failed = _organize_failed
    cls._translation_tab = _translation_tab
    cls.refresh_translation_models = refresh_translation_models
    cls.start_translation = start_translation
    cls.pause_translation = pause_translation
    cls._translation_done = _translation_done
    cls._translation_failed = _translation_failed
    cls._print_translation = _print_translation
