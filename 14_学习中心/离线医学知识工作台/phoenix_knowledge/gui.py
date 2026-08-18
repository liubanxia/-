from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextDocument
from PySide6.QtPrintSupport import QPrintDialog, QPrintPreviewDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .workbench import MedicalKnowledgeWorkbench


class IngestWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, workbench: MedicalKnowledgeWorkbench, files: list[str]):
        super().__init__()
        self.workbench = workbench
        self.files = files

    def run(self):
        try:
            total_files = len(self.files)
            messages = []
            for file_index, filename in enumerate(self.files, start=1):
                def callback(done, total, message):
                    base = int(((file_index - 1) / max(total_files, 1)) * 100)
                    span = 100 / max(total_files, 1)
                    pct = base + int((done / max(total, 1)) * span)
                    self.progress.emit(pct, 100, f"[{file_index}/{total_files}] {message}")

                result = self.workbench.ingest(Path(filename), progress=callback)
                messages.append(
                    f"{Path(filename).name}: {result.pages_indexed}/{result.pages_total} 页"
                )
            self.completed.emit("\n".join(messages))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class AskWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, workbench: MedicalKnowledgeWorkbench, query: str):
        super().__init__()
        self.workbench = workbench
        self.query = query

    def run(self):
        try:
            self.completed.emit(self.workbench.ask(self.query).text)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class OrganizeWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        workbench: MedicalKnowledgeWorkbench,
        title: str = "",
        instruction: str = "",
        task_id: int | None = None,
    ):
        super().__init__()
        self.workbench = workbench
        self.title = title
        self.instruction = instruction
        self.task_id = task_id

    def run(self):
        try:
            callback = lambda done, total, msg: self.progress.emit(done, total, msg)
            if self.task_id is None:
                output, _task_id = self.workbench.organize(
                    self.title,
                    self.instruction,
                    progress=callback,
                )
            else:
                output, _task_id = self.workbench.resume_task(
                    self.task_id,
                    progress=callback,
                )
            self.completed.emit(str(output))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TranslationWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        workbench: MedicalKnowledgeWorkbench,
        pdf_path: str,
        start_page: int,
        target_language: str,
        retry_warning_pages: bool = False,
    ):
        super().__init__()
        self.workbench = workbench
        self.pdf_path = pdf_path
        self.start_page = start_page
        self.target_language = target_language
        self.retry_warning_pages = retry_warning_pages

    def run(self):
        try:
            result = self.workbench.translate_book(
                Path(self.pdf_path),
                start_page=self.start_page,
                target_language=self.target_language,
                retry_warning_pages=self.retry_warning_pages,
                progress=lambda done, total, msg: self.progress.emit(done, total, msg),
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class NotesWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        workbench: MedicalKnowledgeWorkbench,
        text: str,
        title: str,
        instruction: str,
    ):
        super().__init__()
        self.workbench = workbench
        self.text = text
        self.title = title
        self.instruction = instruction

    def run(self):
        try:
            result = self.workbench.organize_txt(
                self.text,
                title=self.title,
                instruction=self.instruction,
                progress=lambda done, total, msg: self.progress.emit(done, total, msg),
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class EmbeddingWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, workbench: MedicalKnowledgeWorkbench):
        super().__init__()
        self.workbench = workbench

    def run(self):
        try:
            count = self.workbench.retriever.embeddings.build_missing()
            self.completed.emit(f"新增 {count} 个向量")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class WorkbenchWindow(QMainWindow):
    PREVIEW_LIMIT = 300_000

    def __init__(self):
        super().__init__()
        self.workbench = MedicalKnowledgeWorkbench()
        self.worker: QThread | None = None
        self.last_organize_path: Path | None = None
        self.last_translation_path: Path | None = None
        self.last_notes_path: Path | None = None

        self.setWindowTitle("Phoenix 离线医学知识工作台")
        self.resize(1100, 780)

        tabs = QTabWidget()
        tabs.addTab(self._library_tab(), "PDF资料库")
        tabs.addTab(self._qa_tab(), "PDF问答")
        tabs.addTab(self._organize_tab(), "多书知识整理")
        tabs.addTab(self._translation_tab(), "整本书翻译")
        tabs.addTab(self._notes_tab(), "TXT笔记整理")
        self.setCentralWidget(tabs)

        self.refresh_library()
        self.refresh_resume_state()
        self.refresh_translation_models()
        self.statusBar().showMessage(self._status_text())

    def _status_text(self) -> str:
        status = self.workbench.status()
        models = status.get("translation_backends") or []
        return (
            f"资料 {status['documents']} 本 | 知识块 {status['chunks']} | "
            f"LLM={status['llm_backend']} | "
            f"Embedding={'READY' if status['embedding_available'] else '未下载'} | "
            f"翻译模型={','.join(models) if models else '未下载'}"
        )

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _library_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("PDF内容只在本机SSD解析和索引；运行阶段不需要互联网。"))

        self.library_list = QListWidget()
        layout.addWidget(self.library_list, 1)

        buttons = QHBoxLayout()
        add_button = QPushButton("导入PDF")
        refresh_button = QPushButton("刷新")
        embedding_button = QPushButton("生成向量索引")
        use_translation_button = QPushButton("选中书→整本翻译")
        buttons.addWidget(add_button)
        buttons.addWidget(refresh_button)
        buttons.addWidget(embedding_button)
        buttons.addWidget(use_translation_button)
        layout.addLayout(buttons)

        self.ingest_progress = QProgressBar()
        self.ingest_progress.setRange(0, 100)
        self.ingest_label = QLabel("等待任务")
        layout.addWidget(self.ingest_progress)
        layout.addWidget(self.ingest_label)

        add_button.clicked.connect(self.add_pdfs)
        refresh_button.clicked.connect(self.refresh_library)
        embedding_button.clicked.connect(self.build_embeddings)
        use_translation_button.clicked.connect(self.use_selected_book_for_translation)
        return widget

    def _qa_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("只根据已导入PDF回答；结论保留书名、页码和来源编号。"))
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText(
            "例如：整理肺磨玻璃结节的CT恶性征象、鉴别诊断和漏诊点，每条保留来源。"
        )
        self.query_edit.setMaximumHeight(150)
        layout.addWidget(self.query_edit)

        row = QHBoxLayout()
        ask_button = QPushButton("根据PDF回答")
        print_preview = QPushButton("打印预览")
        print_button = QPushButton("打印")
        row.addWidget(ask_button)
        row.addWidget(print_preview)
        row.addWidget(print_button)
        layout.addLayout(row)

        self.answer_view = QTextBrowser()
        layout.addWidget(self.answer_view, 1)
        ask_button.clicked.connect(self.ask_question)
        print_preview.clicked.connect(
            lambda: self._print_text("Phoenix PDF问答", self.answer_view.toPlainText(), preview=True)
        )
        print_button.clicked.connect(
            lambda: self._print_text("Phoenix PDF问答", self.answer_view.toPlainText(), preview=False)
        )
        return widget

    def _organize_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.multi_book_info = QLabel("默认从所有已导入PDF中跨书检索、去重和合并。")
        layout.addWidget(self.multi_book_info)

        layout.addWidget(QLabel("专题名称"))
        self.topic_title = QLineEdit()
        self.topic_title.setPlaceholderText("例如：肺磨玻璃结节知识专题")
        layout.addWidget(self.topic_title)

        layout.addWidget(QLabel("整理要求"))
        self.organize_edit = QTextEdit()
        self.organize_edit.setPlaceholderText(
            "例如：汇总全部书籍相关内容；分纯磨玻璃/混合磨玻璃；"
            "列良恶性征象、鉴别、随访、漏诊点、报告模板；来源冲突并列。"
        )
        self.organize_edit.setMaximumHeight(180)
        layout.addWidget(self.organize_edit)

        buttons = QHBoxLayout()
        start_button = QPushButton("整理全部书籍")
        self.resume_button = QPushButton("继续未完成任务")
        save_txt_button = QPushButton("另存TXT")
        preview_button = QPushButton("打印预览")
        print_button = QPushButton("打印")
        for button in (start_button, self.resume_button, save_txt_button, preview_button, print_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.organize_progress = QProgressBar()
        self.organize_progress.setRange(0, 100)
        self.organize_status = QLabel("长期整理按批次保存checkpoint，异常中断后可继续。")
        layout.addWidget(self.organize_progress)
        layout.addWidget(self.organize_status)

        self.organize_result = QTextBrowser()
        layout.addWidget(self.organize_result, 1)

        start_button.clicked.connect(self.start_organize)
        self.resume_button.clicked.connect(self.resume_organize)
        save_txt_button.clicked.connect(
            lambda: self._save_text_dialog(
                self.organize_result.toPlainText(),
                self.topic_title.text().strip() or "多书知识整理",
            )
        )
        preview_button.clicked.connect(
            lambda: self._print_text(
                self.topic_title.text().strip() or "多书知识整理",
                self.organize_result.toPlainText(),
                preview=True,
            )
        )
        print_button.clicked.connect(
            lambda: self._print_text(
                self.topic_title.text().strip() or "多书知识整理",
                self.organize_result.toPlainText(),
                preview=False,
            )
        )
        return widget

    def _translation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(
            QLabel(
                "整本书翻译：默认第1页→最后一页；也可指定起始页。每页保存，自动续翻。"
            )
        )

        file_row = QHBoxLayout()
        self.translation_path = QLineEdit()
        self.translation_path.setPlaceholderText("选择要翻译的医学PDF")
        choose_button = QPushButton("选择PDF")
        file_row.addWidget(self.translation_path, 1)
        file_row.addWidget(choose_button)
        layout.addLayout(file_row)

        options = QHBoxLayout()
        options.addWidget(QLabel("从第几页开始："))
        self.translation_start_page = QSpinBox()
        self.translation_start_page.setRange(1, 999999)
        self.translation_start_page.setValue(1)
        options.addWidget(self.translation_start_page)
        options.addWidget(QLabel("目标语言："))
        self.translation_language = QComboBox()
        self.translation_language.addItems(["中文", "繁体中文", "英文"])
        options.addWidget(self.translation_language)
        options.addStretch(1)
        layout.addLayout(options)

        self.translation_models_label = QLabel("翻译模型：检查中")
        layout.addWidget(self.translation_models_label)

        buttons = QHBoxLayout()
        start_button = QPushButton("开始/继续整本翻译")
        retry_button = QPushButton("重试警告页")
        preview_button = QPushButton("译本打印预览")
        print_button = QPushButton("打印译本")
        for button in (start_button, retry_button, preview_button, print_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.translation_progress = QProgressBar()
        self.translation_progress.setRange(0, 100)
        self.translation_status = QLabel("等待翻译任务")
        layout.addWidget(self.translation_progress)
        layout.addWidget(self.translation_status)

        self.translation_result = QTextBrowser()
        self.translation_result.setPlaceholderText(
            "完成后显示译本路径、已用模型、警告页数量和译本预览。"
        )
        layout.addWidget(self.translation_result, 1)

        choose_button.clicked.connect(self.choose_translation_pdf)
        start_button.clicked.connect(lambda: self.start_translation(False))
        retry_button.clicked.connect(lambda: self.start_translation(True))
        preview_button.clicked.connect(lambda: self._print_translation(preview=True))
        print_button.clicked.connect(lambda: self._print_translation(preview=False))
        return widget

    def _notes_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(
            QLabel("可导入TXT/MD，也可直接粘贴文字。点击一次整理为可保存、可复习、可打印笔记。")
        )

        file_row = QHBoxLayout()
        self.notes_path = QLineEdit()
        self.notes_path.setPlaceholderText("可选：TXT/MD文件")
        choose_button = QPushButton("读取TXT/MD")
        file_row.addWidget(self.notes_path, 1)
        file_row.addWidget(choose_button)
        layout.addLayout(file_row)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("笔记标题："))
        self.notes_title = QLineEdit("医学笔记")
        title_row.addWidget(self.notes_title, 1)
        layout.addLayout(title_row)

        self.notes_source = QTextEdit()
        self.notes_source.setPlaceholderText("粘贴或读取原始TXT笔记……")
        self.notes_source.setMaximumHeight(210)
        layout.addWidget(self.notes_source)

        self.notes_instruction = QLineEdit()
        self.notes_instruction.setPlaceholderText(
            "整理要求（可选）：例如按征象→诊断→鉴别→陷阱→报告表达整理"
        )
        layout.addWidget(self.notes_instruction)

        buttons = QHBoxLayout()
        organize_button = QPushButton("整理笔记")
        save_button = QPushButton("保存TXT")
        preview_button = QPushButton("打印预览")
        print_button = QPushButton("打印")
        for button in (organize_button, save_button, preview_button, print_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.notes_progress = QProgressBar()
        self.notes_progress.setRange(0, 100)
        self.notes_status = QLabel("等待TXT整理")
        layout.addWidget(self.notes_progress)
        layout.addWidget(self.notes_status)

        self.notes_result = QTextBrowser()
        layout.addWidget(self.notes_result, 1)

        choose_button.clicked.connect(self.choose_notes_file)
        organize_button.clicked.connect(self.start_notes_organize)
        save_button.clicked.connect(
            lambda: self._save_text_dialog(
                self.notes_result.toPlainText(),
                self.notes_title.text().strip() or "医学笔记",
            )
        )
        preview_button.clicked.connect(
            lambda: self._print_text(
                self.notes_title.text().strip() or "医学笔记",
                self.notes_result.toPlainText(),
                preview=True,
            )
        )
        print_button.clicked.connect(
            lambda: self._print_text(
                self.notes_title.text().strip() or "医学笔记",
                self.notes_result.toPlainText(),
                preview=False,
            )
        )
        return widget

    def add_pdfs(self):
        if self._busy():
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择医学PDF",
            str(self.workbench.paths.source_root),
            "PDF Files (*.pdf)",
        )
        if not files:
            return
        self.worker = IngestWorker(self.workbench, files)
        self.worker.progress.connect(self._ingest_progress)
        self.worker.completed.connect(self._ingest_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _ingest_progress(self, done: int, total: int, message: str):
        self.ingest_progress.setValue(int(done / max(total, 1) * 100))
        self.ingest_label.setText(message)

    def _ingest_done(self, message: str):
        self.ingest_progress.setValue(100)
        self.ingest_label.setText(message)
        self.refresh_library()

    def refresh_library(self):
        self.library_list.clear()
        documents = self.workbench.db.list_documents()
        for row in documents:
            warning = f" | {row['warning']}" if row["warning"] else ""
            item = QListWidgetItem(
                f"{row['title']} | {row['indexed_pages']}/{row['page_count']}页 | "
                f"{row['status']}{warning}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(row["path"]))
            self.library_list.addItem(item)
        if hasattr(self, "multi_book_info"):
            self.multi_book_info.setText(
                f"当前资料库 {len(documents)} 本。多书整理会从全部已导入PDF中跨书检索、去重、合并。"
            )
        self.statusBar().showMessage(self._status_text())

    def use_selected_book_for_translation(self):
        item = self.library_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.translation_path.setText(str(path))

    def build_embeddings(self):
        if self._busy():
            return
        if not self.workbench.retriever.embeddings.available():
            QMessageBox.information(
                self,
                "Phoenix",
                "Embedding模型尚未下载：\n"
                f"{self.workbench.retriever.embeddings.model_path}",
            )
            return
        self.worker = EmbeddingWorker(self.workbench)
        self.worker.completed.connect(
            lambda msg: QMessageBox.information(self, "Phoenix", msg)
        )
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def ask_question(self):
        if self._busy():
            return
        query = self.query_edit.toPlainText().strip()
        if not query:
            return
        self.answer_view.setPlainText("正在从PDF知识库检索并整理……")
        self.worker = AskWorker(self.workbench, query)
        self.worker.completed.connect(self.answer_view.setPlainText)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _connect_organize_worker(self):
        self.worker.progress.connect(self._organize_progress)
        self.worker.completed.connect(self._organize_done)
        self.worker.failed.connect(self._organize_failed)
        self.worker.start()

    def start_organize(self):
        if self._busy():
            return
        title = self.topic_title.text().strip()
        instruction = self.organize_edit.toPlainText().strip()
        if not instruction:
            return
        self.organize_progress.setValue(0)
        self.organize_result.setPlainText("正在从全部PDF进行跨书检索与长期整理……")
        self.worker = OrganizeWorker(
            self.workbench,
            title=title,
            instruction=instruction,
        )
        self._connect_organize_worker()

    def refresh_resume_state(self):
        if not hasattr(self, "resume_button"):
            return
        task = self.workbench.latest_resumable_task()
        enabled = task is not None and not self._busy()
        self.resume_button.setEnabled(enabled)
        if task is None:
            self.resume_button.setText("没有未完成任务")
            return
        self.resume_button.setText(
            f"继续任务 #{int(task['id'])} ({int(task['progress'])}/{int(task['total'])})"
        )

    def resume_organize(self):
        if self._busy():
            return
        task = self.workbench.latest_resumable_task()
        if task is None:
            self.refresh_resume_state()
            return
        payload = json.loads(task["payload_json"] or "{}")
        self.topic_title.setText(str(payload.get("title", "")))
        self.organize_edit.setPlainText(str(payload.get("instruction", "")))
        total = max(int(task["total"]), 1)
        done = int(task["progress"])
        self.organize_progress.setValue(int(done / total * 100))
        self.organize_status.setText(f"正在从checkpoint继续任务 #{int(task['id'])}……")
        self.organize_result.setPlainText("正在恢复跨书整理任务……")
        self.worker = OrganizeWorker(self.workbench, task_id=int(task["id"]))
        self._connect_organize_worker()

    def _organize_progress(self, done: int, total: int, message: str):
        self.organize_progress.setValue(int(done / max(total, 1) * 100))
        self.organize_status.setText(message)

    def _organize_done(self, output: str):
        self.last_organize_path = Path(output)
        self.organize_progress.setValue(100)
        self.organize_status.setText(f"整理完成：{output}")
        self.organize_result.setPlainText(self._preview_path(Path(output)))
        self.refresh_resume_state()

    def _organize_failed(self, error: str):
        self.organize_status.setText(
            "任务失败；已完成批次checkpoint仍保留，可继续未完成任务。"
        )
        self.refresh_resume_state()
        self._failed(error)

    def choose_translation_pdf(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "选择整本翻译PDF",
            str(self.workbench.paths.source_root),
            "PDF Files (*.pdf)",
        )
        if file:
            self.translation_path.setText(file)

    def refresh_translation_models(self):
        if not hasattr(self, "translation_models_label"):
            return
        models = self.workbench.translator.engine.available_backends()
        self.translation_models_label.setText(
            "当前翻译模型：" + (" → ".join(models) if models else "未下载")
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
        self.translation_result.setPlainText("整本书翻译任务正在运行……")
        self.worker = TranslationWorker(
            self.workbench,
            path,
            self.translation_start_page.value(),
            self.translation_language.currentText(),
            retry_warning_pages=retry_warning_pages,
        )
        self.worker.progress.connect(self._translation_progress)
        self.worker.completed.connect(self._translation_done)
        self.worker.failed.connect(self._translation_failed)
        self.worker.start()

    def _translation_progress(self, done: int, total: int, message: str):
        self.translation_progress.setValue(int(done / max(total, 1) * 100))
        self.translation_status.setText(message)

    def _translation_done(self, result):
        self.last_translation_path = Path(result.output_path)
        self.translation_progress.setValue(100)
        self.translation_status.setText(
            f"整本翻译完成 | 警告页={result.warning_pages} | "
            f"模型={','.join(result.available_backends)}"
        )
        preview = self._preview_path(self.last_translation_path)
        self.translation_result.setPlainText(
            f"完整译本：{self.last_translation_path}\n"
            f"起始页：{result.start_page}\n总页数：{result.total_pages}\n"
            f"续翻跳过页：{result.resumed_pages}\n警告页：{result.warning_pages}\n"
            f"可用模型：{', '.join(result.available_backends)}\n\n{preview}"
        )
        self.refresh_translation_models()

    def _translation_failed(self, error: str):
        self.translation_status.setText("整本翻译失败；已经完成的页面仍保存在checkpoint，可再次继续。")
        self._failed(error)

    def _print_translation(self, preview: bool):
        if self.last_translation_path is None or not self.last_translation_path.is_file():
            QMessageBox.information(self, "Phoenix", "当前没有已完成的整本译本可打印。")
            return
        text = self.last_translation_path.read_text(encoding="utf-8", errors="replace")
        self._print_text(self.last_translation_path.stem, text, preview=preview)

    def choose_notes_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "选择TXT/MD医学笔记",
            str(self.workbench.paths.evidence_root),
            "Text Files (*.txt *.md);;All Files (*)",
        )
        if not file:
            return
        path = Path(file)
        self.notes_path.setText(file)
        self.notes_title.setText(path.stem)
        self.notes_source.setPlainText(
            path.read_text(encoding="utf-8-sig", errors="replace")
        )

    def start_notes_organize(self):
        if self._busy():
            return
        text = self.notes_source.toPlainText().strip()
        if not text:
            return
        self.notes_progress.setValue(0)
        self.notes_status.setText("正在整理TXT医学笔记……")
        self.worker = NotesWorker(
            self.workbench,
            text,
            self.notes_title.text().strip() or "医学笔记",
            self.notes_instruction.text().strip(),
        )
        self.worker.progress.connect(self._notes_progress)
        self.worker.completed.connect(self._notes_done)
        self.worker.failed.connect(self._notes_failed)
        self.worker.start()

    def _notes_progress(self, done: int, total: int, message: str):
        self.notes_progress.setValue(int(done / max(total, 1) * 100))
        self.notes_status.setText(message)

    def _notes_done(self, result):
        self.last_notes_path = Path(result.output_path)
        self.notes_progress.setValue(100)
        self.notes_status.setText(
            f"整理完成 | 模式={result.mode} | 分段={result.chunks} | {result.output_path}"
        )
        self.notes_result.setPlainText(result.text)

    def _notes_failed(self, error: str):
        self.notes_status.setText("TXT笔记整理失败")
        self._failed(error)

    def _preview_path(self, path: Path) -> str:
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) <= self.PREVIEW_LIMIT:
            return text
        half = self.PREVIEW_LIMIT // 2
        return (
            text[:half]
            + "\n\n……（窗口预览已截断；磁盘文件为完整内容）……\n\n"
            + text[-half:]
        )

    def _save_text_dialog(self, text: str, suggested_title: str):
        text = (text or "").strip()
        if not text:
            return
        default = self.workbench.paths.evidence_root / f"{suggested_title}.txt"
        file, _ = QFileDialog.getSaveFileName(
            self,
            "保存TXT",
            str(default),
            "Text Files (*.txt)",
        )
        if file:
            Path(file).write_text(text.rstrip() + "\n", encoding="utf-8")

    def _print_text(self, title: str, text: str, *, preview: bool):
        text = (text or "").strip()
        if not text:
            return
        document = QTextDocument()
        document.setPlainText(text)
        printer = QPrinter()
        printer.setDocName(title)

        if preview:
            dialog = QPrintPreviewDialog(printer, self)
            dialog.setWindowTitle(f"打印预览 - {title}")
            dialog.paintRequested.connect(lambda target: document.print_(target))
            dialog.exec()
            return

        dialog = QPrintDialog(printer, self)
        dialog.setWindowTitle(f"打印 - {title}")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            document.print_(printer)

    def _failed(self, error: str):
        QMessageBox.warning(self, "Phoenix 医学知识工作台", error)

    def closeEvent(self, event):
        if self._busy():
            QMessageBox.information(
                self,
                "Phoenix",
                "当前任务仍在运行。长任务已完成部分会持续保存；为避免当前批次被强制中断，"
                "请等待当前批次完成后关闭。",
            )
            event.ignore()
            return
        self.workbench.close()
        event.accept()


def run_gui() -> int:
    app = QApplication.instance() or QApplication([])
    window = WorkbenchWindow()
    window.show()
    return app.exec()
