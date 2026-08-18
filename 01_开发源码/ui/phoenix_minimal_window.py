from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton,
    QMessageBox,
)

from core.environment_paths import resolve_image_root, resolve_project_root
from output.ai_report_window import AIReportWindow
from output.lesion_button import LesionButton


def _log_path(filename: str) -> Path:
    path = resolve_project_root() / "09_日志" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class CaseLoadWorker(QThread):
    loaded = Signal(object, object)
    failed = Signal(str)

    def __init__(self, root):
        super().__init__()
        self.root = root

    def run(self):
        try:
            from core.yunpacs_live_controller import YUNPACSLiveController

            controller = YUNPACSLiveController(root=self.root)
            controller.poll_once()
            identity = controller.case_identity()

            if not identity:
                raise RuntimeError(
                    "没有读取到可确认的YUNPACS病例"
                )

            self.loaded.emit(controller, identity)

        except Exception as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


class AnalyzeWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def run(self):
        import traceback

        log = _log_path("医院实测_AI执行跟踪.log")

        def write(text):
            with log.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"[{datetime.now()}] {text}\n"
                )

        try:
            write("ANALYZE_START")

            result = self.controller.analyze_current()
            write(f"ANALYZE_RETURN type={type(result).__name__}")

            if isinstance(result, dict):
                selected = result.get("selected_models", []) or []
                execution = result.get("execution_summary", {}) or {}
                diagnostic_executed = bool(
                    result.get("diagnostic_executed", False)
                )
                diagnostic_valid = bool(
                    result.get("diagnostic_valid", False)
                )

                write(f"SELECTED_MODELS={selected}")
                write(f"DIAGNOSTIC_EXECUTED={diagnostic_executed}")
                write(f"DIAGNOSTIC_VALID={diagnostic_valid}")

                analysis = result.get("analysis")
                lesions = (
                    getattr(analysis, "lesions", [],) or []
                    if analysis is not None
                    else []
                )

                write(
                    "DIAGNOSTIC_LESION_COUNT="
                    + (
                        str(len(lesions))
                        if diagnostic_executed
                        else "N/A"
                    )
                )

                for item in execution.get("models", []) or []:
                    if not isinstance(item, dict):
                        continue
                    write(
                        f"MODEL={item.get('model_name')} "
                        f"status={item.get('status')} "
                        f"executed={item.get('executed')} "
                        f"processed={item.get('processed_images')} "
                        f"lesions={item.get('lesion_count')} "
                        f"device={item.get('device', '')} "
                        f"backend={item.get('backend', '')} "
                        f"error={item.get('error', '')}"
                    )

            write("ANALYZE_END")
            self.completed.emit(result)

        except Exception:
            error = traceback.format_exc()
            write("ANALYZE_ERROR\n" + error)
            self.failed.emit(error[-3000:])


class PhoenixMinimalWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Project Phoenix")
        self.resize(680, 300)

        configured_root = os.environ.get(
            "PHOENIX_YUNPACS_ROOT"
        )
        resolved_root = resolve_image_root(
            configured_root
        )
        self.yunpacs_root = str(
            resolved_root
            if resolved_root is not None
            else (
                configured_root
                or "D:/YUNPACS/放射诊断/ImageDir_r"
            )
        )

        self.phase = "WAITING"
        self.live = None
        self.worker = None
        self.last_result = None

        self.report_window = None
        self.lesion_button = LesionButton()

        body = QWidget()
        self.setCentralWidget(body)
        root = QVBoxLayout(body)

        title = QLabel("Project Phoenix")
        title.setStyleSheet(
            "font-size:22px;"
            "font-weight:600;"
            "padding:12px;"
        )
        root.addWidget(title)

        self.case_label = QLabel(
            "当前病例：等待 YUNPACS 病例"
        )
        self.status_label = QLabel(
            "状态：等待医生操作"
        )

        root.addWidget(self.case_label)
        root.addWidget(self.status_label)

        buttons = QHBoxLayout()

        self.ai_button = QPushButton("读取当前病例")
        self.report_button = QPushButton("AI报告")
        self.ai_button.setMinimumHeight(48)
        self.report_button.setMinimumHeight(48)

        buttons.addWidget(self.ai_button)
        buttons.addWidget(self.report_button)
        root.addLayout(buttons)

        self.ai_button.clicked.connect(self.ai_action)
        self.report_button.clicked.connect(self.show_report)

    def ai_action(self):
        if self.worker is not None and self.worker.isRunning():
            return

        if self.phase == "WAITING":
            self.load_case()
            return

        if self.phase == "CONFIRM":
            self.start_analysis()
            return

        if self.phase == "DONE":
            self.reset_for_next_case()
            self.load_case()

    def load_case(self):
        self.ai_button.setEnabled(False)
        self.status_label.setText(
            "状态：正在读取并校验YUNPACS病例..."
        )

        self.worker = CaseLoadWorker(self.yunpacs_root)
        self.worker.loaded.connect(self.case_loaded)
        self.worker.failed.connect(self.worker_failed)
        self.worker.finished.connect(
            lambda: self.ai_button.setEnabled(True)
        )
        self.worker.start()

    def case_loaded(self, controller, identity):
        self.live = controller

        case_id = identity.get("case_id") or "未知"
        modality = ",".join(identity.get("modalities") or [])
        series_count = identity.get("series_count", 0)
        file_count = identity.get("file_count", 0)
        uid = identity.get("study_uid") or "未知"
        path = identity.get("path") or ""
        warnings = identity.get("warnings") or []

        self.case_label.setText(
            f"当前病例：{case_id} | {modality} | "
            f"{series_count}序列 / {file_count}文件"
        )

        tooltip = (
            f"StudyUID: {uid}\n"
            f"路径: {path}"
        )
        if warnings:
            tooltip += "\n\n" + "\n".join(
                f"警告: {warning}"
                for warning in warnings
            )
        self.case_label.setToolTip(tooltip)

        self.status_label.setText(
            "状态：病例已按StudyUID绑定，请核对后开始AI"
        )
        self.ai_button.setText("确认病例并开始AI")
        self.phase = "CONFIRM"

    def start_analysis(self):
        if self.live is None:
            return

        self.ai_button.setEnabled(False)
        self.status_label.setText("状态：AI正在分析...")

        self.worker = AnalyzeWorker(self.live)
        self.worker.completed.connect(self.analysis_finished)
        self.worker.failed.connect(self.worker_failed)
        self.worker.start()

    def analysis_finished(self, result):
        self.last_result = result

        try:
            log = _log_path("医院实测_AI原始结果.log")
            selected = []
            raw = {}
            lesions = []
            execution = {}
            diagnostic_executed = False
            diagnostic_valid = False

            if isinstance(result, dict):
                selected = result.get("selected_models", []) or []
                execution = result.get("execution_summary", {}) or {}
                diagnostic_executed = bool(
                    result.get("diagnostic_executed", False)
                )
                diagnostic_valid = bool(
                    result.get("diagnostic_valid", False)
                )

                analysis = result.get("analysis")
                if analysis is not None:
                    raw = getattr(
                        analysis,
                        "raw_model_results",
                        {},
                    ) or {}
                    lesions = getattr(
                        analysis,
                        "lesions",
                        [],
                    ) or []

            lesion_count_text = (
                str(len(lesions))
                if diagnostic_executed
                else "N/A"
            )

            lines = [
                "",
                "=" * 60,
                f"TIME: {datetime.now()}",
                f"SELECTED_MODELS: {selected}",
                f"DIAGNOSTIC_EXECUTED: {diagnostic_executed}",
                f"DIAGNOSTIC_VALID: {diagnostic_valid}",
                f"DIAGNOSTIC_LESION_COUNT: {lesion_count_text}",
                "MODEL_EXECUTION:",
            ]

            for item in execution.get("models", []) or []:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"  {item.get('model_name')}: "
                    f"status={item.get('status')} "
                    f"executed={item.get('executed')} "
                    f"processed={item.get('processed_images')} "
                    f"lesions={item.get('lesion_count')} "
                    f"error={item.get('error', '')}"
                )

            text = "\n".join(lines)
            print(text, flush=True)

            with log.open("a", encoding="utf-8") as stream:
                stream.write(text + "\n")

        except Exception as exc:
            print(
                "AI DEBUG LOG ERROR:",
                exc,
                flush=True,
            )

        if isinstance(result, dict):
            critical = (
                result.get("execution_summary", {}) or {}
            ).get("critical_incomplete_models", []) or []

            if critical:
                self.status_label.setText(
                    "状态：AI分析未完整完成，请查看报告/日志"
                )
            elif result.get("diagnostic_executed", False):
                self.status_label.setText(
                    "状态：疾病诊断模型已执行，结果待医生复核"
                )
            else:
                self.status_label.setText(
                    "状态：辅助/筛查分析完成，未形成疾病诊断"
                )
        else:
            self.status_label.setText("状态：AI分析完成")

        self.ai_button.setEnabled(True)
        self.ai_button.setText("读取下一病例")
        self.phase = "DONE"

        try:
            memory = self.live.runtime.memory
            if memory is not None and getattr(memory, "images", None):
                self.lesion_button.start(memory)
        except Exception:
            pass

    def worker_failed(self, error):
        self.ai_button.setEnabled(True)
        self.status_label.setText("状态：操作失败")

        QMessageBox.warning(
            self,
            "Phoenix",
            error,
        )

    def report_text(self):
        if not isinstance(self.last_result, dict):
            return ""

        analysis = self.last_result.get("analysis")

        if analysis is not None:
            text = getattr(
                analysis,
                "report_draft",
                None,
            )
            if text:
                return str(text)

        return str(
            self.last_result.get("report_draft", "")
        )

    def show_report(self):
        text = self.report_text()

        if not text:
            QMessageBox.information(
                self,
                "Phoenix AI报告",
                "当前没有AI报告。",
            )
            return

        if self.report_window is None:
            self.report_window = AIReportWindow(parent=self)

        self.report_window.set_report(text)
        self.report_window.show()
        self.report_window.raise_()

    def reset_for_next_case(self):
        try:
            self.lesion_button.close()
        except Exception:
            pass

        try:
            if self.live is not None:
                self.live.close()
        except Exception:
            pass

        self.live = None
        self.last_result = None
        self.phase = "WAITING"
        self.case_label.setText(
            "当前病例：等待 YUNPACS 病例"
        )
        self.ai_button.setText("读取当前病例")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Phoenix",
                "AI正在运行，请等待当前操作结束。",
            )
            event.ignore()
            return

        try:
            if self.live is not None:
                self.live.shutdown()
        except Exception:
            pass

        event.accept()
