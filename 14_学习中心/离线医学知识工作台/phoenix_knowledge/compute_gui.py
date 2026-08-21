from __future__ import annotations

import ipaddress
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

from .provider_hub import provider_choices, provider_spec


_INSTALLED = False
_LOCAL = "__local__"


def _local_or_private_host(host: str) -> bool:
    host = str(host or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_loopback or ip.is_private or ip.is_link_local)
    except ValueError:
        return False


class ComputeSettingsDialog(QDialog):
    def __init__(self, workbench, parent=None):
        super().__init__(parent)
        self.workbench = workbench
        self.gateway = workbench.llm.compute
        self.setWindowTitle("模型与算力")
        self.setMinimumWidth(640)
        self._session_keys: dict[str, str] = {}
        self._last_provider_id = ""

        root = QVBoxLayout(self)
        note = QLabel(
            "选择本机模型或云端模型平台。问答、医学翻译、多资料整理共用同一模型路由。"
            "云端仅用于用户主动导入的学习资料；患者影像、PACS病例和患者信息禁止发送到外部平台。"
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("本机自动（推荐，优先 NVIDIA GPU）", _LOCAL)
        for spec in provider_choices():
            self.mode_combo.addItem(spec.label, spec.id)

        self.provider_note = QLabel()
        self.provider_note.setWordWrap(True)

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("API Key；只保存在当前 Phoenix 运行进程")

        self.allow_remote = QCheckBox("本次运行允许发送当前学习资料文本到所选云端模型")

        self.advanced = QCheckBox("显示高级设置")
        self.remote_url = QLineEdit()
        self.fast_model = QLineEdit()
        self.deep_model = QLineEdit()

        form.addRow("模型平台：", self.mode_combo)
        form.addRow("说明：", self.provider_note)
        form.addRow("API Key：", self.api_key)
        form.addRow("", self.allow_remote)
        form.addRow("", self.advanced)
        form.addRow("API 地址：", self.remote_url)
        form.addRow("智能1模型：", self.fast_model)
        form.addRow("智能2模型：", self.deep_model)
        root.addLayout(form)

        platform_row = QHBoxLayout()
        self.platform_button = QPushButton("打开模型平台")
        self.platform_button.clicked.connect(self._open_platform)
        platform_row.addWidget(self.platform_button)
        platform_row.addStretch(1)
        root.addLayout(platform_row)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        action_row = QHBoxLayout()
        detect_button = QPushButton("刷新算力状态")
        detect_button.clicked.connect(self.refresh_status)
        action_row.addWidget(detect_button)
        action_row.addStretch(1)
        root.addLayout(action_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_button is not None:
            save_button.setText("保存并启用")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.advanced.toggled.connect(self._apply_visibility)
        self.mode_combo.currentIndexChanged.connect(self._provider_changed)

        current = _LOCAL
        if self.gateway.requested_mode() == "remote":
            try:
                current = self.gateway.provider_id()
            except Exception:
                current = "deepseek"
        index = self.mode_combo.findData(current)
        self.mode_combo.setCurrentIndex(max(0, index))
        self.allow_remote.setChecked(self.gateway.remote_allowed())
        self._provider_changed()
        self.refresh_status()

    def _capture_key(self) -> None:
        if self._last_provider_id and self._last_provider_id != _LOCAL:
            self._session_keys[self._last_provider_id] = self.api_key.text().strip()

    def _env_key(self, provider_id: str) -> str:
        spec = provider_spec(provider_id)
        return (
            os.environ.get(f"PHOENIX_PROVIDER_{spec.id.upper()}_API_KEY", "").strip()
            or os.environ.get(spec.api_key_env, "").strip()
        )

    def _provider_changed(self) -> None:
        self._capture_key()
        provider_id = str(self.mode_combo.currentData() or _LOCAL)
        self._last_provider_id = provider_id

        if provider_id == _LOCAL:
            self.provider_note.setText(
                "完全本机运行；自动选择可用 CUDA GPU，没有 CUDA 时回退 CPU。"
            )
            self.api_key.clear()
            self.remote_url.clear()
            self.fast_model.clear()
            self.deep_model.clear()
            self.platform_button.setText("本机模式无需账号")
            self.platform_button.setEnabled(False)
            self._apply_visibility()
            self.refresh_status()
            return

        spec = provider_spec(provider_id)
        try:
            config = self.gateway.provider_config(provider_id)
        except Exception:
            config = {}
        self.provider_note.setText(spec.note)
        self.api_key.setText(
            self._session_keys.get(provider_id, self._env_key(provider_id))
        )
        self.remote_url.setText(str(config.get("base_url") or spec.base_url))
        self.fast_model.setText(str(config.get("fast_model") or spec.fast_model))
        self.deep_model.setText(str(config.get("deep_model") or spec.deep_model))
        self.remote_url.setPlaceholderText(spec.base_url or "https://your-endpoint/v1")
        self.fast_model.setPlaceholderText(spec.fast_model or "填写平台模型 ID")
        self.deep_model.setPlaceholderText(spec.deep_model or "填写平台模型 ID")
        self.platform_button.setText(f"打开 {spec.label} 平台")
        self.platform_button.setEnabled(bool(spec.console_url))
        self._apply_visibility()
        self.refresh_status()

    def _apply_visibility(self) -> None:
        cloud = str(self.mode_combo.currentData() or _LOCAL) != _LOCAL
        advanced = self.advanced.isChecked() and cloud
        self.provider_note.setVisible(True)
        self.api_key.setVisible(cloud)
        self.allow_remote.setVisible(cloud)
        self.advanced.setVisible(cloud)
        for widget in (self.remote_url, self.fast_model, self.deep_model):
            widget.setVisible(advanced)

        form = self.layout().itemAt(1).layout()
        if form is not None:
            for row in (5, 6, 7):
                label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if label_item and label_item.widget():
                    label_item.widget().setVisible(advanced)

    def _open_platform(self) -> None:
        provider_id = str(self.mode_combo.currentData() or _LOCAL)
        if provider_id == _LOCAL:
            return
        spec = provider_spec(provider_id)
        if spec.console_url:
            QDesktopServices.openUrl(QUrl(spec.console_url))

    def refresh_status(self) -> None:
        status = self.gateway.status()
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
        else:
            gpu = "未发现可用 NVIDIA CUDA GPU"

        requested_provider = str(self.mode_combo.currentData() or _LOCAL)
        if requested_provider == _LOCAL:
            target = "本机自动"
        else:
            target = provider_spec(requested_provider).label
        warning = f"\n提示：{status.warning}" if status.warning else ""
        self.status_label.setText(
            f"本机检测：{gpu}\n当前选择：{target}\n当前有效算力：{status.label()}{warning}"
        )

    def save(self) -> None:
        self._capture_key()
        selected = str(self.mode_combo.currentData() or _LOCAL)

        if selected == _LOCAL:
            self.gateway.save_settings(mode="auto")
            os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "auto"
            os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "0"
            os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", None)
            os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_URL", None)
            os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST", None)
            os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP", None)
            self.workbench.llm.reload_compute_config()
            self.accept()
            return

        spec = provider_spec(selected)
        key = self.api_key.text().strip()
        remote_url = self.remote_url.text().strip() or spec.base_url
        fast_model = self.fast_model.text().strip() or spec.fast_model
        deep_model = self.deep_model.text().strip() or spec.deep_model

        if not remote_url:
            QMessageBox.warning(self, "缺少 API 地址", "请填写该平台的 API 地址。")
            return
        if not fast_model:
            QMessageBox.warning(self, "缺少模型", "请填写智能1模型 ID。")
            return
        if not deep_model:
            QMessageBox.warning(self, "缺少模型", "请填写智能2模型 ID。")
            return
        if not self.allow_remote.isChecked():
            QMessageBox.warning(
                self,
                "需要明确授权",
                "请确认本次运行允许发送当前学习资料文本到所选云端模型。",
            )
            return

        # Public providers require a key. A custom loopback/LAN OpenAI endpoint
        # may intentionally run without authentication.
        from urllib.parse import urlparse
        host = (urlparse(remote_url).hostname or "").lower()
        localish = _local_or_private_host(host)
        if not key and not (selected == "custom_openai" and localish):
            QMessageBox.warning(
                self,
                "缺少 API Key",
                f"连接 {spec.label} 需要 API Key。",
            )
            return

        # Clear generic overrides so the provider-specific state is authoritative.
        os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", None)
        os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_URL", None)
        os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST", None)
        os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP", None)
        self.gateway.set_provider_api_key(selected, key)
        self.gateway.select_provider(
            selected,
            base_url=remote_url,
            fast_model=fast_model,
            deep_model=deep_model,
        )
        os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "remote"
        os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "1"
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
            if status.requested_mode == "remote":
                try:
                    provider = self.workbench.llm.compute.provider_label()
                    text = f"模型：{provider}"
                except Exception:
                    text = f"算力：{status.label()}"
            else:
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
        try:
            self.statusBar().showMessage(self._status_text())
        except Exception:
            pass

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.compute_status_label = QLabel()
        self.compute_status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        compute_button = QPushButton("模型/算力")
        compute_button.setToolTip("选择本机或多个云端模型平台")
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

    try:
        from .release_gui_truth import install as install_release_gui_truth
        install_release_gui_truth(gui_module)
    except Exception:
        pass
