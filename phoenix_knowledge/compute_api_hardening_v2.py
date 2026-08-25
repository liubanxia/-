from __future__ import annotations

"""Harden the model/API settings dialog against stale probes and QThread crashes.

The original dialog had two production hazards:

1. ``真实测试算力/API`` called ``workbench.llm.generate`` directly.  Therefore a
   newly pasted API key, URL or model was *not* being tested until the dialog had
   already been saved.  To a user this looked like the new key did nothing.
2. The probe worker could block in urllib for the normal remote request timeout
   (up to 180 seconds by default), while Save/Cancel/window-close remained
   available.  Destroying a dialog that still owns a running QThread can abort a
   Qt process (``QThread: Destroyed while thread is still running``).

This layer makes remote probing staged, bounded and lifetime-safe without ever
writing the API key to disk.  It also makes remote consent atomic: both the
knowledge-workbench permission and the medical-translation API-fallback flag are
set before the dialog is accepted.
"""

import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox

from .provider_hub import (
    _chat_url,
    _extract_anthropic_text,
    _extract_openai_text,
    provider_spec,
)


_INSTALLED = False
_LOCAL = "__local__"
_KNOWLEDGE_REMOTE_FLAG = "PHOENIX_KNOWLEDGE_ALLOW_REMOTE"
_TRANSLATION_API_FLAG = "PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"
_PROBE_TIMEOUT_ENV = "PHOENIX_KNOWLEDGE_API_PROBE_TIMEOUT"


@dataclass(frozen=True)
class ProbeSnapshot:
    provider_id: str
    base_url: str
    model: str
    api_key: str


def _probe_timeout() -> int:
    """Return a short settings-dialog timeout, independent of translation jobs."""

    try:
        value = int(os.environ.get(_PROBE_TIMEOUT_ENV, "25") or 25)
    except (TypeError, ValueError):
        value = 25
    return max(5, min(value, 45))


def _local_or_private_host(host: str) -> bool:
    host = str(host or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
        return bool(address.is_loopback or address.is_private or address.is_link_local)
    except ValueError:
        return False


def _snapshot_from_dialog(dialog) -> ProbeSnapshot:
    provider_id = str(dialog.mode_combo.currentData() or _LOCAL).strip()
    if provider_id == _LOCAL:
        raise ValueError("当前选择的是本机模式，不需要 API Key。")

    spec = provider_spec(provider_id)
    base_url = str(dialog.remote_url.text() or "").strip() or spec.base_url
    model = str(dialog.deep_model.text() or "").strip() or spec.deep_model
    api_key = str(dialog.api_key.text() or "").strip()

    if not base_url:
        raise ValueError("请填写 API 地址。")
    if not model:
        raise ValueError("请填写模型 ID。")

    host = (urllib.parse.urlparse(base_url).hostname or "").strip().lower()
    localish = _local_or_private_host(host)
    if not api_key and not (provider_id == "custom_openai" and localish):
        raise ValueError(f"连接 {spec.label} 需要 API Key。")

    return ProbeSnapshot(
        provider_id=provider_id,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )


def _probe_remote(snapshot: ProbeSnapshot) -> str:
    """Probe exactly the values currently typed in the dialog.

    This function deliberately does not touch ComputeGateway, environment
    variables, provider_hub.json or compute_gateway.json.  The pasted key stays
    only in the worker snapshot for the lifetime of this process.
    """

    spec = provider_spec(snapshot.provider_id)
    url = _chat_url(snapshot.base_url, spec.protocol)
    if not url:
        raise RuntimeError(f"{spec.label} API 地址为空")

    prompt = "只回复：Phoenix API 测试通过。"
    if spec.protocol == "anthropic":
        payload = {
            "model": snapshot.model,
            "max_tokens": 16,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if snapshot.api_key:
            headers["x-api-key"] = snapshot.api_key
    else:
        payload = {
            "model": snapshot.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 16,
            "stream": False,
        }
        # A settings probe should be cheap and deterministic.  Do not enable
        # reasoning just because the configured production model is a deep one.
        if spec.id in {"deepseek", "zhipu"}:
            payload["thinking"] = {"type": "disabled"}
        elif spec.id in {"qwen", "siliconflow"}:
            payload["enable_thinking"] = False
        elif spec.id == "gemini":
            payload["reasoning_effort"] = "low"

        headers = {"Content-Type": "application/json"}
        if snapshot.api_key:
            headers["Authorization"] = f"Bearer {snapshot.api_key}"
        if spec.id == "openrouter":
            headers["X-Title"] = "Phoenix Medical Knowledge Workbench"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = _probe_timeout()
    started = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        if code in {401, 403}:
            raise RuntimeError(
                f"{spec.label} API Key 验证失败（HTTP {code}）。"
                "请确认 Key、平台和账号一致。"
            ) from exc
        if code == 404:
            raise RuntimeError(
                f"{spec.label} API 地址或模型接口不存在（HTTP 404）。"
                "请检查 API 地址和模型 ID。"
            ) from exc
        raise RuntimeError(
            f"{spec.label} API 返回 HTTP {code or '错误'}；"
            "请检查模型 ID、额度和平台设置。"
        ) from exc
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc) or exc)
        raise RuntimeError(
            f"{spec.label} 网络连接失败：{reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"{spec.label} API 在 {timeout} 秒内没有响应。"
        ) from exc
    except Exception as exc:
        # Never include request headers or the key in diagnostics.
        raise RuntimeError(
            f"{spec.label} API 测试失败：{type(exc).__name__}: {exc}"
        ) from exc

    if spec.protocol == "anthropic":
        text = _extract_anthropic_text(data)
    else:
        text = _extract_openai_text(data)

    elapsed = time.perf_counter() - started
    return (
        f"{spec.label} API 测试通过（{elapsed:.2f}s） · "
        f"模型={snapshot.model} · {text[:80]}"
    )


class _SafeRemoteProbeWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, snapshot: ProbeSnapshot, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot

    def run(self) -> None:
        try:
            self.completed.emit(_probe_remote(self.snapshot))
        except Exception as exc:
            self.failed.emit(str(exc))


def _probe_running(dialog) -> bool:
    worker = getattr(dialog, "_probe_worker", None)
    if worker is None:
        return False
    try:
        return bool(worker.isRunning())
    except RuntimeError:
        return False


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def install(gui_module) -> None:
    """Install the final lifetime-safe API settings contract."""

    del gui_module
    global _INSTALLED
    if _INSTALLED:
        return

    from .compute_gui import ComputeSettingsDialog

    original_test_compute = ComputeSettingsDialog.test_compute
    original_reject = ComputeSettingsDialog.reject
    original_close_event = ComputeSettingsDialog.closeEvent

    def _set_probe_message(self, text: str) -> None:
        try:
            self.status_label.setText(str(text))
        except Exception:
            pass

    def _remote_probe_done(self, message: str) -> None:
        try:
            self.probe_button.setEnabled(True)
        except Exception:
            pass
        _set_probe_message(self, message)

    def _remote_probe_failed(self, error: str) -> None:
        try:
            self.probe_button.setEnabled(True)
        except Exception:
            pass
        _set_probe_message(self, "真实测试失败：" + str(error))

    def _remote_probe_finished(self) -> None:
        worker = getattr(self, "_probe_worker", None)
        if worker is not None:
            try:
                worker.deleteLater()
            except Exception:
                pass
        self._probe_worker = None
        try:
            self.probe_button.setEnabled(True)
        except Exception:
            pass

    def test_compute(self) -> None:
        if _probe_running(self):
            _set_probe_message(self, "API/算力测试仍在运行，请等待当前测试结束。")
            return

        selected = str(self.mode_combo.currentData() or _LOCAL).strip()
        if selected == _LOCAL:
            # Preserve the existing local-GPU/model probe, but the close/reject
            # guards below now also protect its QThread lifetime.
            return original_test_compute(self)

        try:
            snapshot = _snapshot_from_dialog(self)
        except ValueError as exc:
            _set_probe_message(self, str(exc))
            QMessageBox.warning(self, "API 设置不完整", str(exc))
            return

        spec = provider_spec(snapshot.provider_id)
        timeout = _probe_timeout()
        self.probe_button.setEnabled(False)
        _set_probe_message(
            self,
            f"正在验证 {spec.label} 当前输入的 API Key/地址/模型……"
            f"最长等待 {timeout} 秒。",
        )
        self._probe_worker = _SafeRemoteProbeWorker(snapshot, parent=self)
        self._probe_worker.completed.connect(self._remote_probe_done)
        self._probe_worker.failed.connect(self._remote_probe_failed)
        self._probe_worker.finished.connect(self._remote_probe_finished)
        self._probe_worker.start()

    def reject(self) -> None:
        if _probe_running(self):
            _set_probe_message(
                self,
                "API/算力测试正在运行。为防止后台线程被销毁导致程序闪退，"
                "测试结束后即可关闭此窗口。",
            )
            return
        original_reject(self)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if _probe_running(self):
            _set_probe_message(
                self,
                "API/算力测试正在运行。已阻止关闭窗口，避免 QThread 闪退；"
                "测试结束后可立即关闭。",
            )
            event.ignore()
            return
        original_close_event(self, event)

    def save(self) -> None:
        if _probe_running(self):
            _set_probe_message(self, "请等待当前 API/算力测试结束后再保存。")
            return

        self._capture_key()
        selected = str(self.mode_combo.currentData() or _LOCAL).strip()

        if selected == _LOCAL:
            old_knowledge = os.environ.get(_KNOWLEDGE_REMOTE_FLAG)
            old_translation = os.environ.get(_TRANSLATION_API_FLAG)
            try:
                self.gateway.save_settings(mode="auto")
                os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "auto"
                os.environ[_KNOWLEDGE_REMOTE_FLAG] = "0"
                os.environ[_TRANSLATION_API_FLAG] = "0"
                os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", None)
                os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_URL", None)
                os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST", None)
                os.environ.pop("PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP", None)
                self.workbench.llm.reload_compute_config()
            except Exception as exc:
                _restore_env(_KNOWLEDGE_REMOTE_FLAG, old_knowledge)
                _restore_env(_TRANSLATION_API_FLAG, old_translation)
                message = f"本机模式保存失败：{type(exc).__name__}: {exc}"
                _set_probe_message(self, message)
                QMessageBox.critical(self, "模型/算力设置保存失败", message)
                return
            self.accept()
            return

        try:
            snapshot = _snapshot_from_dialog(self)
        except ValueError as exc:
            _set_probe_message(self, str(exc))
            QMessageBox.warning(self, "API 设置不完整", str(exc))
            return

        if not self.allow_remote.isChecked():
            message = "请确认本次运行允许发送当前学习资料文本到所选云端模型。"
            _set_probe_message(self, message)
            QMessageBox.warning(self, "需要明确授权", message)
            return

        spec = provider_spec(snapshot.provider_id)
        provider_key_env = f"PHOENIX_PROVIDER_{spec.id.upper()}_API_KEY"
        old_values = {
            _KNOWLEDGE_REMOTE_FLAG: os.environ.get(_KNOWLEDGE_REMOTE_FLAG),
            _TRANSLATION_API_FLAG: os.environ.get(_TRANSLATION_API_FLAG),
            "PHOENIX_KNOWLEDGE_ACCELERATOR": os.environ.get(
                "PHOENIX_KNOWLEDGE_ACCELERATOR"
            ),
            provider_key_env: os.environ.get(provider_key_env),
        }

        try:
            # Keys remain process-only. select_provider persists URL/model/provider
            # metadata, but never the key itself.
            self.gateway.set_provider_api_key(snapshot.provider_id, snapshot.api_key)
            self.gateway.select_provider(
                snapshot.provider_id,
                base_url=snapshot.base_url,
                fast_model=snapshot.model,
                deep_model=snapshot.model,
            )
            os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "remote"
            os.environ[_KNOWLEDGE_REMOTE_FLAG] = "1"
            # Set translation consent BEFORE accept().  This removes the old
            # post-close wrapper race and keeps both remote gates consistent.
            os.environ[_TRANSLATION_API_FLAG] = "1"
            self.workbench.llm.reload_compute_config()
        except Exception as exc:
            for name, value in old_values.items():
                _restore_env(name, value)
            message = (
                f"{spec.label} 设置未启用：{type(exc).__name__}: {exc}。"
                "窗口保持打开，可修改后重试。"
            )
            _set_probe_message(self, message)
            QMessageBox.critical(self, "模型/API 设置保存失败", message)
            return

        _set_probe_message(self, f"{spec.label} 已保存并启用。")
        self.accept()

    ComputeSettingsDialog._remote_probe_done = _remote_probe_done
    ComputeSettingsDialog._remote_probe_failed = _remote_probe_failed
    ComputeSettingsDialog._remote_probe_finished = _remote_probe_finished
    ComputeSettingsDialog.test_compute = test_compute
    ComputeSettingsDialog.reject = reject
    ComputeSettingsDialog.closeEvent = closeEvent
    ComputeSettingsDialog.save = save
    ComputeSettingsDialog.__phoenix_api_dialog_hardening__ = 2

    _INSTALLED = True
