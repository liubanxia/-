from __future__ import annotations

import time

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QMessageBox

from .gui_enhancements import _OrganizeWorkerV2, _TranslationWorkerV2

_INSTALLED = False


class _EmbeddingWorkerV2(QThread):
    progress = Signal(int, int, str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, workbench):
        super().__init__()
        self.workbench = workbench

    def run(self):
        try:
            state = self.workbench.retriever.embeddings.readiness()
            total = max(0, int(state.get("missing", 0) or 0))
            if total <= 0:
                self.completed.emit("语义索引已经完整，无需重建。")
                return
            self.progress.emit(
                0,
                total,
                "正在加载本地语义模型，准备补齐缺失向量……",
            )

            def callback(done, _total, message):
                self.progress.emit(
                    min(max(0, int(done)), total),
                    total,
                    str(message),
                )

            added = self.workbench.retriever.embeddings.build_missing(
                progress=callback,
            )
            final = self.workbench.retriever.embeddings.readiness()
            if int(final.get("missing", 0) or 0) != 0:
                raise RuntimeError(
                    str(final.get("label") or "语义索引仍不完整")
                )
            self.completed.emit(
                f"语义索引完成：新增 {added} 个向量；"
                f"{final.get('label', '')}"
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _elapsed_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{sec:02d}秒"
    if minutes:
        return f"{minutes}分{sec:02d}秒"
    return f"{sec}秒"


def install(gui_module) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow

    original_init = cls.__init__
    original_status_text = cls._status_text
    original_ingest_done = cls._ingest_done
    original_organize_progress = cls._organize_progress
    original_translation_progress = cls._translation_progress
    original_start_organize = cls.start_organize
    original_resume_organize = cls.resume_organize
    original_organize_paused = cls._organize_paused
    original_organize_done = cls._organize_done
    original_organize_failed = cls._organize_failed
    original_start_translation = cls.start_translation
    original_pause_translation = cls.pause_translation
    original_translation_done = cls._translation_done
    original_translation_failed = cls._translation_failed
    original_refresh_translation_models = cls.refresh_translation_models

    def _status_text(self) -> str:
        try:
            status = self.workbench.status()
            smart1 = (
                "READY"
                if self.workbench.llm.available("fast")
                else "未就绪"
            )
            smart2 = (
                "READY"
                if self.workbench.llm.available("deep")
                else "未就绪"
            )
            semantic = str(
                status.get("semantic_label") or "语义状态未知"
            )
            return (
                f"资料 {status['documents']} 本 | 知识块 {status['chunks']} | "
                f"{semantic} | 智能1={smart1} | 智能2={smart2}"
            )
        except Exception:
            return original_status_text(self)

    def _heartbeat_tick(self) -> None:
        worker = getattr(self, "worker", None)
        now = time.monotonic()
        if isinstance(worker, _OrganizeWorkerV2) and worker.isRunning():
            started = getattr(self, "_organize_started_at", now)
            last_at = getattr(self, "_organize_last_at", started)
            last_msg = getattr(
                self,
                "_organize_last_msg",
                "正在执行多资料整理",
            )
            self.organize_status.setText(
                f"{last_msg} | 已运行 {_elapsed_text(now - started)} | "
                f"最近响应 {_elapsed_text(now - last_at)}前"
            )
        elif isinstance(worker, _TranslationWorkerV2) and worker.isRunning():
            started = getattr(self, "_translation_started_at", now)
            last_at = getattr(self, "_translation_last_at", started)
            last_msg = getattr(
                self,
                "_translation_last_msg",
                "正在执行整本翻译",
            )
            self.translation_status.setText(
                f"{last_msg} | 已运行 {_elapsed_text(now - started)} | "
                f"最近响应 {_elapsed_text(now - last_at)}前"
            )
        elif isinstance(worker, _EmbeddingWorkerV2) and worker.isRunning():
            started = getattr(self, "_embedding_started_at", now)
            last_at = getattr(self, "_embedding_last_at", started)
            last_msg = getattr(
                self,
                "_embedding_last_msg",
                "正在建立语义索引",
            )
            self.ingest_label.setText(
                f"{last_msg} | 已运行 {_elapsed_text(now - started)} | "
                f"最近响应 {_elapsed_text(now - last_at)}前"
            )

    def _maybe_auto_index(self) -> None:
        if self._busy():
            return
        try:
            status = self.workbench.status()
        except Exception:
            return
        if (
            status.get("embedding_model_ready")
            and status.get("embedding_runtime_available")
            and int(status.get("embedding_missing", 0) or 0) > 0
        ):
            self.build_embeddings()

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._translation_resume_pending = False
        self._organize_resume_pending = False
        self._organize_started_at = 0.0
        self._organize_last_at = 0.0
        self._organize_last_msg = ""
        self._translation_started_at = 0.0
        self._translation_last_at = 0.0
        self._translation_last_msg = ""
        self._embedding_started_at = 0.0
        self._embedding_last_at = 0.0
        self._embedding_last_msg = ""

        self._phoenix_heartbeat_timer = QTimer(self)
        self._phoenix_heartbeat_timer.setInterval(1000)
        self._phoenix_heartbeat_timer.timeout.connect(
            self._heartbeat_tick
        )
        self._phoenix_heartbeat_timer.start()

        QTimer.singleShot(350, self._maybe_auto_index)
        self.statusBar().showMessage(self._status_text())

    def _ingest_done(self, message: str):
        result = original_ingest_done(self, message)
        # GUI batch import intentionally defers vector generation. Run exactly
        # once after all successful files have committed their chunks.
        QTimer.singleShot(100, self._maybe_auto_index)
        return result

    def _embedding_progress(
        self,
        done: int,
        total: int,
        message: str,
    ):
        self._embedding_last_at = time.monotonic()
        self._embedding_last_msg = str(message)
        self.ingest_progress.setValue(
            int(done / max(total, 1) * 100)
        )
        self.ingest_label.setText(str(message))

    def _embedding_done(self, message: str):
        self.ingest_progress.setValue(100)
        self.ingest_label.setText(str(message))
        self.statusBar().showMessage(self._status_text())

    def build_embeddings(self):
        if self._busy():
            return
        try:
            status = self.workbench.status()
        except Exception as exc:
            self._failed(f"{type(exc).__name__}: {exc}")
            return

        if not status.get("embedding_model_ready"):
            QMessageBox.information(
                self,
                "Phoenix 语义检索",
                "本地 Embedding 模型尚未准备好。基础关键词检索仍可使用。",
            )
            return
        if not status.get("embedding_runtime_available"):
            QMessageBox.information(
                self,
                "Phoenix 语义检索",
                "语义运行组件尚未准备好。"
                "请使用 Phoenix 启动器执行一次环境自检。",
            )
            return

        missing = max(
            0,
            int(status.get("embedding_missing", 0) or 0),
        )
        if missing == 0:
            self.ingest_progress.setValue(100)
            self.ingest_label.setText(
                str(
                    status.get("semantic_label")
                    or "语义索引已完整"
                )
            )
            self.statusBar().showMessage(self._status_text())
            return

        now = time.monotonic()
        self._embedding_started_at = now
        self._embedding_last_at = now
        self._embedding_last_msg = (
            f"准备补齐 {missing} 个语义向量"
        )
        self.ingest_progress.setValue(0)
        self.ingest_label.setText(self._embedding_last_msg)
        self.worker = _EmbeddingWorkerV2(self.workbench)
        self.worker.progress.connect(self._embedding_progress)
        self.worker.completed.connect(self._embedding_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _organize_progress(
        self,
        done: int,
        total: int,
        message: str,
    ):
        self._organize_last_at = time.monotonic()
        self._organize_last_msg = str(message)
        return original_organize_progress(
            self,
            done,
            total,
            message,
        )

    def _translation_progress(
        self,
        done: int,
        total: int,
        message: str,
    ):
        self._translation_last_at = time.monotonic()
        self._translation_last_msg = str(message)
        return original_translation_progress(
            self,
            done,
            total,
            message,
        )

    def start_organize(self):
        if self._busy():
            worker = getattr(self, "worker", None)
            if (
                isinstance(worker, _OrganizeWorkerV2)
                and worker.isRunning()
                and worker._pause_requested.is_set()
            ):
                self._organize_resume_pending = True
                self._organize_last_at = time.monotonic()
                self._organize_last_msg = (
                    "已收到继续请求；当前批次安全暂停后将自动继续"
                )
                self.organize_status.setText(
                    self._organize_last_msg
                )
            return
        self._organize_resume_pending = False
        now = time.monotonic()
        self._organize_started_at = now
        self._organize_last_at = now
        self._organize_last_msg = "正在启动多资料整理"
        return original_start_organize(self)

    def resume_organize(self):
        if self._busy():
            worker = getattr(self, "worker", None)
            if (
                isinstance(worker, _OrganizeWorkerV2)
                and worker.isRunning()
                and worker._pause_requested.is_set()
            ):
                self._organize_resume_pending = True
                self._organize_last_at = time.monotonic()
                self._organize_last_msg = (
                    "已收到继续请求；完成当前批次并暂停后将自动恢复"
                )
                self.organize_status.setText(
                    self._organize_last_msg
                )
            return
        now = time.monotonic()
        self._organize_started_at = now
        self._organize_last_at = now
        self._organize_last_msg = "正在恢复多资料整理"
        return original_resume_organize(self)

    def pause_organize(self):
        worker = getattr(self, "worker", None)
        if isinstance(worker, _OrganizeWorkerV2) and worker.isRunning():
            worker.request_pause()
            self._organize_resume_pending = False
            self._organize_last_at = time.monotonic()
            self._organize_last_msg = (
                "已请求暂停；完成当前批次后安全停止"
            )
            self.organize_status.setText(
                self._organize_last_msg
            )
            if hasattr(self, "pause_organize_button"):
                self.pause_organize_button.setEnabled(False)

    def _organize_paused(self, task_id: int):
        result = original_organize_paused(self, task_id)
        if self._organize_resume_pending:
            self._organize_resume_pending = False
            self.organize_status.setText(
                f"任务 #{int(task_id)} 已安全暂停，正在自动继续……"
            )
            QTimer.singleShot(250, self.resume_organize)
        return result

    def _organize_done(self, output: str):
        self._organize_resume_pending = False
        return original_organize_done(self, output)

    def _organize_failed(self, error: str):
        self._organize_resume_pending = False
        return original_organize_failed(self, error)

    def refresh_translation_models(self):
        try:
            smart1 = (
                "可用"
                if self.workbench.llm.available("fast")
                else "未就绪"
            )
            smart2 = (
                "可用"
                if self.workbench.llm.available("deep")
                else "未就绪"
            )
            names = list(
                self.workbench.translator.engine.available_backends()
            )
            fallback_names = [
                name
                for name in names
                if "qwen" not in name.lower()
            ]
            fallback = "可用" if fallback_names else "未就绪"
            commercial = bool(
                self.workbench.status().get("commercial_release")
            )
            suffix = (
                " | 商业版已禁用非商业模型"
                if commercial
                else ""
            )
            self.translation_models_label.setText(
                f"翻译能力：智能1={smart1} | "
                f"智能2={smart2} | 自动兜底={fallback}{suffix}"
            )
        except Exception:
            return original_refresh_translation_models(self)

    def start_translation(self, retry_warning_pages: bool):
        worker = getattr(self, "worker", None)
        if isinstance(worker, _TranslationWorkerV2) and worker.isRunning():
            if worker._pause_requested.is_set():
                self._translation_resume_pending = True
                self._translation_last_at = time.monotonic()
                self._translation_last_msg = (
                    "已收到继续请求；当前页安全保存并暂停后将自动续翻"
                )
                self.translation_status.setText(
                    self._translation_last_msg
                )
            else:
                self._translation_last_at = time.monotonic()
                self._translation_last_msg = (
                    "当前翻译仍在运行；无需重复点击"
                )
                self.translation_status.setText(
                    self._translation_last_msg
                )
            return
        if self._busy():
            return

        self._translation_resume_pending = False
        now = time.monotonic()
        self._translation_started_at = now
        self._translation_last_at = now
        self._translation_last_msg = "正在启动整本医学翻译"
        return original_start_translation(
            self,
            retry_warning_pages,
        )

    def pause_translation(self):
        self._translation_resume_pending = False
        worker = getattr(self, "worker", None)
        if isinstance(worker, _TranslationWorkerV2) and worker.isRunning():
            self._translation_last_at = time.monotonic()
            self._translation_last_msg = (
                "已请求暂停；完成当前页后安全停止"
            )
        return original_pause_translation(self)

    def _translation_done(self, result):
        paused = bool(getattr(result, "paused", False))
        pending = bool(
            getattr(self, "_translation_resume_pending", False)
        )
        result_value = original_translation_done(self, result)
        if paused and pending:
            self._translation_resume_pending = False
            self.translation_status.setText(
                "当前页已安全保存；正在自动继续整本翻译……"
            )
            QTimer.singleShot(
                300,
                lambda: self.start_translation(False),
            )
        return result_value

    def _translation_failed(self, error: str):
        self._translation_resume_pending = False
        return original_translation_failed(self, error)

    cls._status_text = _status_text
    cls._heartbeat_tick = _heartbeat_tick
    cls._maybe_auto_index = _maybe_auto_index
    cls.__init__ = _init
    cls._ingest_done = _ingest_done
    cls._embedding_progress = _embedding_progress
    cls._embedding_done = _embedding_done
    cls.build_embeddings = build_embeddings
    cls._organize_progress = _organize_progress
    cls._translation_progress = _translation_progress
    cls.start_organize = start_organize
    cls.resume_organize = resume_organize
    cls.pause_organize = pause_organize
    cls._organize_paused = _organize_paused
    cls._organize_done = _organize_done
    cls._organize_failed = _organize_failed
    cls.refresh_translation_models = refresh_translation_models
    cls.start_translation = start_translation
    cls.pause_translation = pause_translation
    cls._translation_done = _translation_done
    cls._translation_failed = _translation_failed
