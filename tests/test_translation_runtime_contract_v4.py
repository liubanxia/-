from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from phoenix_knowledge import translation_runtime_contract_v4 as v4
from phoenix_knowledge.translation_models import MultiModelTranslationEngine


class _Compute:
    def __init__(self):
        self.key = "session-secret"

    def requested_mode(self):
        return "remote"

    def status(self):
        return SimpleNamespace(
            effective_mode="remote",
            remote_allowed=True,
        )

    def remote_model(self, profile=None):
        return "deepseek-v4-pro"

    def remote_is_public(self):
        return True

    def remote_api_key(self):
        return self.key

    def provider_label(self):
        return "DeepSeek"


class _LLM:
    def __init__(self):
        self.compute = _Compute()

    def available(self, profile=None):
        return profile == "translation"


class _Qwen:
    name = "qwen35_medical_translation"

    def __init__(self):
        self.llm = _LLM()


class TranslationRuntimeContractV4Tests(unittest.TestCase):
    def setUp(self):
        v4._INSTALLED = False

    def test_remote_compute_ready_is_translation_api_ready_even_if_legacy_flag_was_stale(self):
        engine = SimpleNamespace(qwen=_Qwen())
        with patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_TRANSLATION_ALLOW_API_FALLBACK": "0",
            },
            clear=False,
        ):
            self.assertTrue(v4._remote_api_ready(engine))
            self.assertEqual(
                os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"],
                "1",
            )

    def test_missing_key_cannot_be_reported_ready_for_public_provider(self):
        qwen = _Qwen()
        qwen.llm.compute.key = ""
        engine = SimpleNamespace(qwen=qwen)
        self.assertFalse(v4._remote_api_ready(engine))

    def test_v4_replaces_smart2_only_preflight_inventory_with_full_chain_inventory(self):
        from phoenix_knowledge import translation_cascade_v2 as cascade

        ready = {
            "model1_ready": False,
            "model1_names": (),
            "model2_ready": False,
            "model2_path": "",
            "model3_ready": False,
            "model3_path": "",
            "api_ready": True,
            "formal_ready": True,
        }
        with patch.object(v4, "chain_status", return_value=ready):
            v4.install()
            engine = object.__new__(MultiModelTranslationEngine)
            engine.qwen = _Qwen()

            names = engine.formal_backend_names("中文")
            active = engine.active_backends("中文", "smart2")

            self.assertTrue(active)
            self.assertEqual(len(active), 1)
            self.assertEqual(names, ["API:DeepSeek/deepseek-v4-pro"])
            self.assertIs(
                cascade._run_local_cascade,
                __import__(
                    "phoenix_knowledge.translation_chain_enforcement_v3",
                    fromlist=["_run_quality_chain"],
                )._run_quality_chain,
            )

    def test_local_model3_alone_is_enough_to_pass_formal_preflight_inventory(self):
        ready = {
            "model1_ready": False,
            "model1_names": (),
            "model2_ready": False,
            "model2_path": "",
            "model3_ready": True,
            "model3_path": "D:/models/qwen3",
            "api_ready": False,
            "formal_ready": True,
        }
        with patch.object(v4, "chain_status", return_value=ready):
            v4.install()
            engine = object.__new__(MultiModelTranslationEngine)
            engine.qwen = _Qwen()
            names = engine.formal_backend_names("中文")
            active = engine.active_backends("中文", "smart2")

        self.assertTrue(active)
        self.assertEqual(names, ["M3:Qwen-local-medical"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
