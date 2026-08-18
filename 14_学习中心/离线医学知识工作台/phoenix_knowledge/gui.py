from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
                    base = int(
                        ((file_index - 1) / max(total_files, 1)) * 100
                    )
                    span = 100 / max(total_files, 1)
                    pct = base + int(
                        (done / max(total, 1)) * span
                    )
                    self.progress.emit(
                        pct,
                        100,
                        f"[{file_index}/{total_files}] {message}",
                    )

                result = self.workbench.ingest(
                    Path(filename),
                    progress=callback,
                )
                messages.append(
                    f"{Path(filename).name}: "
                    f"{result.pages_indexed}/{result.pages_total} 页"
                )
            self.completed.emit("\n".join(messages))
        except Exception as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


class AskWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        workbench: MedicalKnowledgeWorkbench,
        query: str,
    ):
        super().__init__()
        self.workbench = workbench
        self.query = query

    def run(self):
        try:
            self.completed.emit(
                self.workbench.ask(self.query).text
            )
        except Exception as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


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
            callback = lambda done, total, msg: self.progress.emit(
                done,
                total,
                msg,
            )
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
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


class WorkbenchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.workbench = MedicalKnowledgeWorkbench()
        self.worker = None
        self.setWindowTitle(
            "Phoenix 离线医学知识工作台"
        )
        self.resize(980, 720)

        tabs = QTabWidget()
        tabs.addTab(
            self._library_tab(),
            "PDF资料库",
        )
        tabs.addTab(
            self._qa_tab(),
            "问答",
        )
        tabs.addTab(
            self._organize_tab(),
            "深度整理",
        )
        self.setCentralWidget(tabs)
        self.statusBar().showMessage(
            self._status_text()
        )
        self.refresh_library()
        self.refresh_resume_state()

    def _status_text(self) -> str:
        status = self.workbench.status()
        return (
            f"资料 {status['documents']} 本 | "
            f"知识块 {status['chunks']} | "
            f"LLM={status['llm_backend']} | "
            f"Embedding="
            f"{'READY' if status['embedding_available'] else '未下载'}"
        )

    def _library_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(
            QLabel(
                "PDF内容只在本机SSD解析和索引；运行时不需要互联网。"
            )
        )

        self.library_list = QListWidget()
        layout.addWidget(self.library_list, 1)

        buttons = QHBoxLayout()
        add_button = QPushButton("导入PDF")
        refresh_button = QPushButton("刷新")
        embedding_button = QPushButton(
            "生成向量索引"
        )
        buttons.addWidget(add_button)
        buttons.addWidget(refresh_button)
        buttons.addWidget(embedding_button)
        layout.addLayout(buttons)

        self.ingest_progress = QProgressBar()
        self.ingest_progress.setRange(0, 100)
        self.ingest_label = QLabel("等待任务")
        layout.addWidget(self.ingest_progress)
        layout.addWidget(self.ingest_label)

        add_button.clicked.connect(self.add_pdfs)
        refresh_button.clicked.connect(
            self.refresh_library
        )
        embedding_button.clicked.connect(
            self.build_embeddings
        )
        return widget

    def _qa_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(
            QLabel(
                "输入你要从已导入PDF中查询/整理的问题："
            )
        )
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText(
            "例如：整理肺磨玻璃结节的CT恶性征象、鉴别诊断和漏诊点，每条保留来源。"
        )
        self.query_edit.setMaximumHeight(140)
        layout.addWidget(self.query_edit)
        ask_button = QPushButton("根据PDF回答")
        layout.addWidget(ask_button)
        self.answer_view = QTextBrowser()
        layout.addWidget(self.answer_view, 1)
        ask_button.clicked.connect(
            self.ask_question
        )
        return widget

    def _organize_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("专题名称"))
        self.topic_title = QLineEdit()
        self.topic_title.setPlaceholderText(
            "例如：肺磨玻璃结节知识专题"
        )
        layout.addWidget(self.topic_title)
        layout.addWidget(QLabel("整理要求"))
        self.organize_edit = QTextEdit()
        self.organize_edit.setPlaceholderText(
            "例如：汇总所有PDF相关内容；分纯磨玻璃/混合磨玻璃；"
            "列良恶性征象、鉴别、随访、漏诊点、报告模板；冲突来源并列。"
        )
        layout.addWidget(self.organize_edit)

        organize_buttons = QHBoxLayout()
        start_button = QPushButton(
            "开始长期整理"
        )
        self.resume_button = QPushButton(
            "继续未完成任务"
        )
        organize_buttons.addWidget(start_button)
        organize_buttons.addWidget(self.resume_button)
        layout.addLayout(organize_buttons)

        self.organize_progress = QProgressBar()
        self.organize_progress.setRange(0, 100)
        self.organize_status = QLabel(
            "任务会分批保存checkpoint；异常退出后可继续最近未完成任务。"
        )
        layout.addWidget(self.organize_progress)
        layout.addWidget(self.organize_status)
        self.organize_result = QTextBrowser()
        layout.addWidget(self.organize_result, 1)
        start_button.clicked.connect(
            self.start_organize
        )
        self.resume_button.clicked.connect(
            self.resume_organize
        )
        return widget

    def _busy(self) -> bool:
        return (
            self.worker is not None
            and self.worker.isRunning()
        )

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
        self.worker = IngestWorker(
            self.workbench,
            files,
        )
        self.worker.progress.connect(
            self._ingest_progress
        )
        self.worker.completed.connect(
            self._ingest_done
        )
        self.worker.failed.connect(
            self._failed
        )
        self.worker.start()

    def _ingest_progress(
        self,
        done: int,
        total: int,
        message: str,
    ):
        self.ingest_progress.setValue(
            int(done / max(total, 1) * 100)
        )
        self.ingest_label.setText(message)

    def _ingest_done(self, message: str):
        self.ingest_progress.setValue(100)
        self.ingest_label.setText(message)
        self.refresh_library()

    def refresh_library(self):
        self.library_list.clear()
        for row in self.workbench.db.list_documents():
            warning = (
                f" | {row['warning']}"
                if row["warning"]
                else ""
            )
            self.library_list.addItem(
                f"{row['title']} | "
                f"{row['indexed_pages']}/{row['page_count']}页 | "
                f"{row['status']}{warning}"
            )
        self.statusBar().showMessage(
            self._status_text()
        )

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

        class EmbeddingWorker(QThread):
            completed = Signal(str)
            failed = Signal(str)

            def __init__(self, workbench):
                super().__init__()
                self.workbench = workbench

            def run(self):
                try:
                    count = (
                        self.workbench
                        .retriever
                        .embeddings
                        .build_missing()
                    )
                    self.completed.emit(
                        f"新增 {count} 个向量"
                    )
                except Exception as exc:
                    self.failed.emit(
                        f"{type(exc).__name__}: {exc}"
                    )

        self.worker = EmbeddingWorker(
            self.workbench
        )
        self.worker.completed.connect(
            lambda msg: QMessageBox.information(
                self,
                "Phoenix",
                msg,
            )
        )
        self.worker.failed.connect(
            self._failed
        )
        self.worker.start()

    def ask_question(self):
        if self._busy():
            return
        query = self.query_edit.toPlainText().strip()
        if not query:
            return
        self.answer_view.setPlainText(
            "正在从PDF知识库检索并整理……"
        )
        self.worker = AskWorker(
            self.workbench,
            query,
        )
        self.worker.completed.connect(
            self.answer_view.setPlainText
        )
        self.worker.failed.connect(
            self._failed
        )
        self.worker.start()

    def _connect_organize_worker(self):
        self.worker.progress.connect(
            self._organize_progress
        )
        self.worker.completed.connect(
            self._organize_done
        )
        self.worker.failed.connect(
            self._organize_failed
        )
        self.worker.start()

    def start_organize(self):
        if self._busy():
            return
        title = self.topic_title.text().strip()
        instruction = (
            self.organize_edit
            .toPlainText()
            .strip()
        )
        if not instruction:
            return
        self.organize_progress.setValue(0)
        self.organize_result.setPlainText(
            "长期整理任务正在运行……"
        )
        self.worker = OrganizeWorker(
            self.workbench,
            title=title,
            instruction=instruction,
        )
        self._connect_organize_worker()

    def refresh_resume_state(self):
        task = self.workbench.latest_resumable_task()
        enabled = task is not None and not self._busy()
        self.resume_button.setEnabled(enabled)
        if task is None:
            self.resume_button.setText(
                "没有未完成任务"
            )
            return

        self.resume_button.setText(
            f"继续任务 #{int(task['id'])} "
            f"({int(task['progress'])}/{int(task['total'])})"
        )

    def resume_organize(self):
        if self._busy():
            return
        task = self.workbench.latest_resumable_task()
        if task is None:
            self.refresh_resume_state()
            return

        payload = json.loads(
            task["payload_json"] or "{}"
        )
        self.topic_title.setText(
            str(payload.get("title", ""))
        )
        self.organize_edit.setPlainText(
            str(payload.get("instruction", ""))
        )
        total = max(int(task["total"]), 1)
        done = int(task["progress"])
        self.organize_progress.setValue(
            int(done / total * 100)
        )
        self.organize_status.setText(
            f"正在从 checkpoint 继续任务 #{int(task['id'])}……"
        )
        self.organize_result.setPlainText(
            "正在恢复长期整理任务……"
        )
        self.worker = OrganizeWorker(
            self.workbench,
            task_id=int(task["id"]),
        )
        self._connect_organize_worker()

    def _organize_progress(
        self,
        done: int,
        total: int,
        message: str,
    ):
        self.organize_progress.setValue(
            int(done / max(total, 1) * 100)
        )
        self.organize_status.setText(message)

    def _organize_done(self, output: str):
        self.organize_progress.setValue(100)
        self.organize_status.setText(
            "整理完成"
        )
        self.organize_result.setPlainText(
            f"已保存：\n{output}"
        )
        self.refresh_resume_state()

    def _organize_failed(self, error: str):
        self.organize_status.setText(
            "任务失败；已完成批次的checkpoint仍保留，可点击继续未完成任务。"
        )
        self.refresh_resume_state()
        self._failed(error)

    def _failed(self, error: str):
        QMessageBox.warning(
            self,
            "Phoenix 医学知识工作台",
            error,
        )

    def closeEvent(self, event):
        if self._busy():
            QMessageBox.information(
                self,
                "Phoenix",
                "当前任务仍在运行。已完成批次会持续写入checkpoint；"
                "为避免当前批次被强行中断，请等待本批完成后关闭。",
            )
            event.ignore()
            return
        self.workbench.close()
        event.accept()


def run_gui() -> int:
    app = (
        QApplication.instance()
        or QApplication([])
    )
    window = WorkbenchWindow()
    window.show()
    return app.exec()
