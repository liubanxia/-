from __future__ import annotations

import json
import os
import types
import unittest
from unittest.mock import patch

from phoenix_knowledge import compute_api_hardening_v2 as hardening
from phoenix_knowledge.compute_gui import ComputeSettingsDialog


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _Text:
    def __init__(self, value: str):
        self.value = value

    def text(self):
        return self.value


class _Combo:
    def __init__(self, value: str):
        self.value = value

    def currentData(self):
        return self.value


class _Check:
    def __init__(self, value: bool):
        self.value = value

    def isChecked(self):
        return self.value


class _Label:
    def __init__(self):
        self.value = ""

    def setText(self, text: str):
        self.value = str(text)


class _Gateway:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.key_calls = []
        self.select_calls = []

    def set_provider_api_key(self, provider_id: str, key: str):
        self.key_calls.append((provider_id, key))

    def select_provider(self, provider_id: str, **kwargs):
        self.select_calls.append((provider_id, kwargs))
        if self.fail:
            raise RuntimeError("simulated settings failure")

    def save_settings(self, **kwargs):
        if self.fail:
            raise RuntimeError("simulated settings failure")


class _LLM:
    def __init__(self):
        self.reload_count = 0

    def reload_compute_config(self):
        self.reload_count += 1


class _DialogDouble:
    def __init__(self, *, fail: bool = False):
        self.mode_combo = _Combo("deepseek")
        self.remote_url = _Text("https://staged.example/v1")
        self.deep_model = _Text("staged-medical-model")
        self.api_key = _Text("staged-secret-key")
        self.allow_remote = _Check(True)
        self.status_label = _Label()
        self.gateway = _Gateway(fail=fail)
        self.workbench = types.SimpleNamespace(llm=_LLM())
        self.accepted = False
        self.flags_seen_at_accept = None
        self._probe_worker = None

    def _capture_key(self):
        return None

    def accept(self):
        self.flags_seen_at_accept = (
            os.environ.get("PHOENIX_KNOWLEDGE_ALLOW_REMOTE"),
            os.environ.get("PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"),
        )
        self.accepted = True


class _RunningWorker:
    def isRunning(self):
        return True


class _CloseEvent:
    def __init__(self):
        self.ignored = False

    def ignore(self):
        self.ignored = True


class ComputeAPIHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardening.install(None)

    def test_remote_probe_uses_current_staged_key_url_and_model(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response(
                {"choices": [{"message": {"content": "Phoenix API 测试通过。"}}]}
            )

        snapshot = hardening.ProbeSnapshot(
            provider_id="deepseek",
            base_url="https://staged.example/v1",
            model="staged-medical-model",
            api_key="staged-secret-key",
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = hardening._probe_remote(snapshot)

        self.assertEqual(
            captured["url"],
            "https://staged.example/v1/chat/completions",
        )
        self.assertEqual(captured["authorization"], "Bearer staged-secret-key")
        self.assertEqual(captured["payload"]["model"], "staged-medical-model")
        self.assertEqual(captured["timeout"], 25)
        self.assertIn("测试通过", result)

    def test_probe_timeout_is_short_and_bounded_independently(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["timeout"] = timeout
            return _Response(
                {"choices": [{"message": {"content": "ok"}}]}
            )

        snapshot = hardening.ProbeSnapshot(
            provider_id="deepseek",
            base_url="https://staged.example/v1",
            model="m",
            api_key="k",
        )
        with patch.dict(
            os.environ,
            {"PHOENIX_KNOWLEDGE_API_PROBE_TIMEOUT": "999"},
            clear=False,
        ), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            hardening._probe_remote(snapshot)

        self.assertEqual(captured["timeout"], 45)

    def test_save_enables_both_remote_gates_before_accept(self):
        dialog = _DialogDouble()
        with patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "0",
                "PHOENIX_TRANSLATION_ALLOW_API_FALLBACK": "0",
            },
            clear=False,
        ):
            ComputeSettingsDialog.save(dialog)
            self.assertTrue(dialog.accepted)
            self.assertEqual(dialog.flags_seen_at_accept, ("1", "1"))
            self.assertEqual(dialog.workbench.llm.reload_count, 1)
            self.assertEqual(
                dialog.gateway.key_calls,
                [("deepseek", "staged-secret-key")],
            )
            self.assertEqual(
                dialog.gateway.select_calls[0][1]["deep_model"],
                "staged-medical-model",
            )

    def test_save_failure_keeps_dialog_open_and_does_not_raise(self):
        dialog = _DialogDouble(fail=True)
        with patch.object(hardening.QMessageBox, "critical", return_value=None):
            ComputeSettingsDialog.save(dialog)
        self.assertFalse(dialog.accepted)
        self.assertIn("设置未启用", dialog.status_label.value)

    def test_running_probe_blocks_reject_and_window_close(self):
        dialog = _DialogDouble()
        dialog._probe_worker = _RunningWorker()

        ComputeSettingsDialog.reject(dialog)
        self.assertIn("闪退", dialog.status_label.value)

        event = _CloseEvent()
        ComputeSettingsDialog.closeEvent(dialog, event)
        self.assertTrue(event.ignored)
        self.assertIn("QThread", dialog.status_label.value)

    def test_hardening_replaces_post_accept_consent_contract(self):
        self.assertEqual(
            getattr(ComputeSettingsDialog, "__phoenix_api_dialog_hardening__", 0),
            2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
