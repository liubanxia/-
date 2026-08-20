from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


_INSTALLED = False


class ComputeSettingsDialog(QDialog):
    def __init__(self, workbench, parent=None):
        super().__init__(parent)
        self.workbench = workbench
        self.gateway = workbench.llm.compute
        self.setWindowTitle("算力设置")
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        note = QLabel(
            "默认完全本地运行。DeepSpeed 仅用于本机GPU加速；外接GPU/API模式会把当前"
            "问答、整理或翻译片段发送到你填写的服务。患者资料禁止使用外接模式。"
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("自动（CPU / 本机GPU）", "auto")
        self.mode_combo.addItem("CPU兼容", "cpu")
        self.mode_combo.addItem("本机CUDA", "cuda")
        self.mode_combo.addItem("本机GPU加速（DeepSpeed）", "deepspeed")
        self.mode_combo.addItem("外接GPU / API服务", "remote")

        self.remote_url = QLineEdit()
        self.remote_url.setPlaceholderText(
            "例如 https://api.deepseek.com 或 http://192.168.1.20:8000/v1"
        )
        self.fast_model = QLineEdit()
        self.fast_model.setPlaceholderText("快速模型；DeepSeek可留空自动使用 v4-flash")
        self.deep_model = QLineEdit()
        self.deep_model.setPlaceholderText("质量模型；DeepSeek可留空自动使用 v4-pro")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("仅保存在当前运行进程，不写入磁盘")
        self.allow_remote = QCheckBox("本次运行允许把文本发送到外接服务")

        settings = self.gateway._settings
        current_mode = self.gateway.requested_mode()
        index = self.mode_combo.findData(current_mode)
        self.mode_combo.setCurrentIndex(max(0, index))
        self.remote_url.setText(settings.remote_url)
        self.fast_model.setText(settings.remote_model_fast)
        self.deep_model.setText(settings.remote_model_deep)
        self.api_key.setText(os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", ""))
        self.allow_remote.setChecked(self.gateway.remote_allowed())

        form.addRow("算力来源：", self.mode_combo)
        form.addRow("外接服务地址：", self.remote_url)
        form.addRow("快速模型：", self.fast_model)
        form.addRow("质量模型：", self.deep_model)
        form.addRow("API密钥：", self.api_key)
        form.addRow("外接授权：", self.allow_remote)
        root.addLayout(form)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        action_row = QHBoxLayout()
        detect_button = QPushButton("检测算力")
        detect_button.clicked.connect(self.refresh_status)
        action_row.addWidget(detect_button)
        action_row.addStretch(1)
        root.addLayout(action_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.refresh_status()

    def refresh_status(self) -> None:
        status = self.gateway.status()
        gpu = "无"
        if status.gpu_names:
            parts = []
            for index, name in enumerate(status.gpu_names):
                vram = (
                    status.gpu_vram_gb[index]
                    if index < len(status.gpu_vram_gb)
                    else 0.0
                )
                parts.append(f"{name} ({vram:.1f}GB)" if vram else name)
            gpu = " / ".join(parts)
        ds = "可用" if status.deepspeed_available else "未安装/不可用"
        warning = f"\n提示：{status.warning}" if status.warning else ""
        self.status_label.setText(
            f"当前：{status.label()} | CUDA GPU：{gpu} | DeepSpeed：{ds}{warning}"
        )

    def save(self) -> None:
        mode = str(self.mode_combo.currentData() or "auto")
        remote_url = self.remote_url.text().strip()
        if mode == "remote":
            if not remote_url:
                QMessageBox.warning(self, "未配置", "外接GPU/API模式必须填写服务地址。")
                return
            if not self.allow_remote.isChecked():
                QMessageBox.warning(
                    self,
                    "需要明确授权",
                    "外接模式会发送当前处理文本。请勾选“本次运行允许把文本发送到外接服务”。",
                )
                return

        self.gateway.save_settings(
            mode=mode,
            remote_url=remote_url,
            remote_model_fast=self.fast_model.text().strip(),
            remote_model_deep=self.deep_model.text().strip(),
        )
        os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = mode
        os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = (
            "1" if mode == "remote" and self.allow_remote.isChecked() else "0"
        )
        key = self.api_key.text().strip()
        if key:
            os.environ["PHOENIX_KNOWLEDGE_REMOTE_API_KEY"] = key
        else:
            os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", None)
        self.workbench.llm.reload_compute_config()
        self.accept()


def install(gui_module) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow
    original_init = cls.__init__

    def _update_compute_label(self):
        try:
            status = self.workbench.llm.compute.status()
            text = f"算力：{status.label()}"
            if status.warning:
                text += " ⚠"
            self.compute_status_label.setText(text)
            self.compute_status_label.setToolTip(status.warning or text)
        except Exception as exc:
            self.compute_status_label.setText("算力：检测失败")
            self.compute_status_label.setToolTip(f"{type(exc).__name__}: {exc}")

    def _open_compute_settings(self):
        dialog = ComputeSettingsDialog(self.workbench, self)
        dialog.exec()
        self._update_compute_label()
        try:
            self.refresh_translation_models()
        except Exception:
            pass

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.compute_status_label = QLabel()
        self.compute_status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        compute_button = QPushButton("算力设置")
        compute_button.setToolTip("选择CPU、本机CUDA、DeepSpeed或外接GPU/API")
        compute_button.clicked.connect(self._open_compute_settings)
        self.statusBar().addPermanentWidget(self.compute_status_label)
        self.statusBar().addPermanentWidget(compute_button)
        self._update_compute_label()

    cls._update_compute_label = _update_compute_label
    cls._open_compute_settings = _open_compute_settings
    cls.__init__ = _init
