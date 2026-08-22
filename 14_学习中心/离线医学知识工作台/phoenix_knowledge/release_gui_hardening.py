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
    """Install the final GUI task/runtime contract.

    This installer must run after every compatibility/product GUI extension.
    It owns all long-task entry points so later modules cannot reintroduce
    silent ``if busy: return`` behaviour.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow

    original_init = cls.__init__
    original_status_text = cls._status_text
    original_ingest_done = cls._ingest_done
    original_add_pdfs = cls.add_pdfs
    original_add_documents = getattr(cls, "add_documents", original_add_pdfs)
    original_ask_question = cls.ask_question
    original_start_notes_organize = cls.start_notes_organize
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

    def _current_task_label(self) -> str:
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning():
            return ""
        if isinstance(worker, _TranslationWorkerV2):
            return "整本翻译"
        if isinstance(worker, _OrganizeWorkerV2):
            return "多资料整理"
        if isinstance(worker, _EmbeddingWorkerV2):
            return "语义索引"
        name = type(worker).__name__.lower()
        if "ingest" in name:
            return "资料导入"
        if "ask" in name:
            return "资料问答"
        if "note" in name:
            return "笔记整理"
        return "后台任务"

    def _busy_notice(self, requested_action: str) -> None:
        current = self._current_task_label() or "后台任务"
        message = (
            f"当前正在执行“{current}”，暂不能同时启动“{requested_action}”。\n\n"
            "任务仍在运行，不是按钮失效。请在对应页查看进度；"
            "整理/翻译任务可使用暂停按钮，在安全保存点暂停后再启动其他任务。"
        )
        try:
            self.statusBar().showMessage(
                f"当前任务：{current}；{requested_action}未启动",
                8000,
            )
        except Exception:
            pass
        QMessageBox.information(
            self,
            "Phoenix 当前有任务运行",
            message,
        )

    def _track_worker(self, label: str) -> None:
        worker = getattr(self, "worker", None)
        if worker is None or not hasattr(worker, "finished"):
            return
        if bool(getattr(worker, "_phoenix_finish_hooked", False)):
            return
        worker._phoenix_finish_hooked = True

        def finished() -> None:
            if getattr(self, "worker", None) is worker:
                self.worker = None
            try:
                self.statusBar().showMessage(
                    f"{label}任务已结束，可以继续使用其他功能。",
                    6000,
                )
            except Exception:
                pass
            try:
                self.refresh_resume_state()
            except Exception:
                pass

        worker.finished.connect(finished)

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

        if hasattr(self, "translation_smart_combo"):
            smart2 = self.translation_smart_combo.findData("smart2")
            if smart2 >= 0:
                self.translation_smart_combo.setItemText(
                    smart2,
                    "医学精译（质量模型，低推理）",
                )

        self._phoenix_heartbeat_timer = QTimer(self)
        self._phoenix_heartbeat_timer.setInterval(1000)
        self._phoenix_heartbeat_timer.timeout.connect(
            self._heartbeat_tick
        )
        self._phoenix_heartbeat_timer.start()

        QTimer.singleShot(350, self._maybe_auto_index)
        self.statusBar().showMessage(self._status_text())

    def add_documents(self):
        if self._busy():
            self._busy_notice("资料导入")
            return
        result = original_add_documents(self)
        self._track_worker("资料导入")
        return result

    def ask_question(self):
        if self._busy():
            self._busy_notice("资料问答")
            return
        result = original_ask_question(self)
        self._track_worker("资料问答")
        return result

    def start_notes_organize(self):
        if self._busy():
            self._busy_notice("笔记整理")
            return
        result = original_start_notes_organize(self)
        self._track_worker("笔记整理")
        return result

    def _ingest_done(self, message: str):
        result = original_ingest_done(self, message)
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
            self._busy_notice("语义索引")
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
        self._track_worker("语义索引")

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
            else:
                self._busy_notice("多资料整理")
            return
        self._organize_resume_pending = False
        now = time.monotonic()
        self._organize_started_at = now
        self._organize_last_at = now
        self._organize_last_msg = "正在启动多资料整理"
        result = original_start_organize(self)
        self._track_worker("多资料整理")
        return result

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
            else:
                self._busy_notice("继续多资料整理")
            return
        now = time.monotonic()
        self._organize_started_at = now
        self._organize_last_at = now
        self._organize_last_msg = "正在恢复多资料整理"
        result = original_resume_organize(self)
        self._track_worker("多资料整理")
        return result

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
            status = self.workbench.status()
            quality = (
                "可用"
                if status.get(
                    "generator_deep_ready",
                    self.workbench.llm.available("translation"),
                )
                else "未就绪"
            )
            engine = self.workbench.translator.engine
            preview = (
                "可用"
                if engine.marian.available() or engine.nllb.available()
                else "未就绪"
            )
            commercial = bool(status.get("commercial_release"))
            suffix = (
                " | 商业版已禁用非商业模型"
                if commercial
                else ""
            )
            self.translation_models_label.setText(
                f"医学精译={quality}（仅质量模型） | "
                f"普通资料快速预览={preview}{suffix}"
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
            self._busy_notice("整本翻译")
            return

        self._translation_resume_pending = False
        now = time.monotonic()
        self._translation_started_at = now
        self._translation_last_at = now
        self._translation_last_msg = "正在启动整本医学翻译"
        result = original_start_translation(
            self,
            retry_warning_pages,
        )
        self._track_worker("整本翻译")
        return result

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

    for guarded in (
        add_documents,
        ask_question,
        start_notes_organize,
        build_embeddings,
        start_organize,
        resume_organize,
        start_translation,
    ):
        guarded.__phoenix_busy_guard__ = True

    cls._status_text = _status_text
    cls._current_task_label = _current_task_label
    cls._busy_notice = _busy_notice
    cls._track_worker = _track_worker
    cls._heartbeat_tick = _heartbeat_tick
    cls._maybe_auto_index = _maybe_auto_index
    cls.__init__ = _init
    cls.add_pdfs = add_documents
    cls.add_documents = add_documents
    cls.ask_question = ask_question
    cls.start_notes_organize = start_notes_organize
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
    cls.__phoenix_release_gui_hardening__ = 2
