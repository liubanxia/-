from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
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
_DEEPSEEK_URL = "https://api.deepseek.com"
_DEEPSEEK_PLATFORM = "https://platform.deepseek.com/"


class ComputeSettingsDialog(QDialog):
    def __init__(self, workbench, parent=None):
        super().__init__(parent)
        self.workbench = workbench
        self.gateway = workbench.llm.compute
        self.setWindowTitle("算力设置")
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)
        note = QLabel(
            "普通使用只需要选择“本机自动”或“DeepSeek 云算力”。"
            "本机自动会优先使用可用的 NVIDIA GPU，没有可用GPU时自动回退CPU。"
            "云算力只允许处理用户主动导入的学习资料；患者数据禁止上传。"
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("本机自动（推荐）", "local")
        self.mode_combo.addItem("DeepSeek 云算力", "deepseek")

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("DeepSeek API Key；仅保存在当前运行进程")
        self.allow_remote = QCheckBox("本次运行允许发送当前学习资料文本到 DeepSeek")

        self.advanced = QCheckBox("显示高级设置")
        self.remote_url = QLineEdit()
        self.remote_url.setPlaceholderText(_DEEPSEEK_URL)
        self.fast_model = QLineEdit()
        self.fast_model.setPlaceholderText("默认 deepseek-v4-flash")
        self.deep_model = QLineEdit()
        self.deep_model.setPlaceholderText("默认 deepseek-v4-pro")

        settings = self.gateway._settings
        requested = self.gateway.requested_mode()
        self.mode_combo.setCurrentIndex(1 if requested == "remote" and self.gateway.is_deepseek_remote() else 0)
        self.remote_url.setText(settings.remote_url or _DEEPSEEK_URL)
        self.fast_model.setText(settings.remote_model_fast)
        self.deep_model.setText(settings.remote_model_deep)
        self.api_key.setText(os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", ""))
        self.allow_remote.setChecked(self.gateway.remote_allowed())

        form.addRow("算力来源：", self.mode_combo)
        form.addRow("API密钥：", self.api_key)
        form.addRow("", self.allow_remote)
        form.addRow("", self.advanced)
        form.addRow("服务地址：", self.remote_url)
        form.addRow("快速模型：", self.fast_model)
        form.addRow("质量模型：", self.deep_model)
        root.addLayout(form)

        platform_row = QHBoxLayout()
        platform_button = QPushButton("打开 DeepSeek API 平台")
        platform_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(_DEEPSEEK_PLATFORM))
        )
        platform_row.addWidget(platform_button)
        platform_row.addStretch(1)
        root.addLayout(platform_row)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        action_row = QHBoxLayout()
        detect_button = QPushButton("检测本机算力")
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

        self.advanced.toggled.connect(self._apply_visibility)
        self.mode_combo.currentIndexChanged.connect(self._apply_visibility)
        self._apply_visibility()
        self.refresh_status()

    def _apply_visibility(self) -> None:
        cloud = self.mode_combo.currentData() == "deepseek"
        advanced = self.advanced.isChecked() and cloud

        self.api_key.setVisible(cloud)
        self.allow_remote.setVisible(cloud)
        self.advanced.setVisible(cloud)

        for widget in (self.remote_url, self.fast_model, self.deep_model):
            widget.setVisible(advanced)

        form = self.layout().itemAt(1).layout()
        if form is not None:
            for row in (4, 5, 6):
                label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if label_item and label_item.widget():
                    label_item.widget().setVisible(advanced)

    def refresh_status(self) -> None:
        status = self.gateway.status()
        if status.gpu_names:
            parts = []
            for index, name in enumerate(status.gpu_names):
                vram = status.gpu_vram_gb[index] if index < len(status.gpu_vram_gb) else 0.0
                parts.append(f"{name} ({vram:.1f}GB)" if vram else name)
            gpu = " / ".join(parts)
        else:
            gpu = "未发现可用 NVIDIA CUDA GPU"

        warning = f"\n提示：{status.warning}" if status.warning else ""
        self.status_label.setText(
            f"本机检测：{gpu}\n当前有效算力：{status.label()}{warning}"
        )

    def save(self) -> None:
        selected = str(self.mode_combo.currentData() or "local")
        if selected == "deepseek":
            key = self.api_key.text().strip()
            if not key:
                QMessageBox.warning(
                    self,
                    "缺少API密钥",
                    "连接 DeepSeek 云算力需要 API Key。",
                )
                return
            if not self.allow_remote.isChecked():
                QMessageBox.warning(
                    self,
                    "需要明确授权",
                    "请确认本次运行允许发送当前学习资料文本到 DeepSeek。",
                )
                return

            remote_url = self.remote_url.text().strip() or _DEEPSEEK_URL
            self.gateway.save_settings(
                mode="remote",
                remote_url=remote_url,
                remote_model_fast=self.fast_model.text().strip(),
                remote_model_deep=self.deep_model.text().strip(),
            )
            os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "remote"
            os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "1"
            os.environ["PHOENIX_KNOWLEDGE_REMOTE_API_KEY"] = key
        else:
            self.gateway.save_settings(
                mode="auto",
                remote_url=self.remote_url.text().strip() or _DEEPSEEK_URL,
                remote_model_fast=self.fast_model.text().strip(),
                remote_model_deep=self.deep_model.text().strip(),
            )
            os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "auto"
            os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "0"
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
        compute_button.setToolTip("普通用户只需选择本机自动或 DeepSeek 云算力")
        compute_button.clicked.connect(self._open_compute_settings)
        self.statusBar().addPermanentWidget(self.compute_status_label)
        self.statusBar().addPermanentWidget(compute_button)
        self._update_compute_label()

    cls._update_compute_label = _update_compute_label
    cls._open_compute_settings = _open_compute_settings
    cls.__init__ = _init

    try:
        from .release_gui_hardening import install as install_release_gui_hardening
        install_release_gui_hardening(gui_module)
    except Exception:
        pass
