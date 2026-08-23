from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.compute_gateway import ComputeGateway
from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.llm import LocalLLM


def _paths(root: Path) -> WorkbenchPaths:
    return WorkbenchPaths(
        project_root=root,
        source_root=root / "sources",
        runtime_root=root / "runtime",
        evidence_root=root / "evidence",
        model_root=root / "models",
        database=root / "runtime" / "knowledge.sqlite3",
        structure_root=root / "runtime" / "structure",
    ).ensure()


class ComputeGatewayTests(unittest.TestCase):
    def test_remote_mode_never_activates_without_explicit_session_authorization(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_REMOTE_URL": "https://api.deepseek.com",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "0",
            },
            clear=False,
        ):
            gateway = ComputeGateway(_paths(Path(temp)))
            status = gateway.status()
            self.assertEqual(status.requested_mode, "remote")
            self.assertNotEqual(status.effective_mode, "remote")
            self.assertFalse(status.remote_allowed)
            self.assertIn("授权", status.warning)

    def test_deepseek_defaults_use_current_fast_and_quality_model_names(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_REMOTE_URL": "https://api.deepseek.com",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
            },
            clear=False,
        ):
            gateway = ComputeGateway(_paths(Path(temp)))
            self.assertEqual(gateway.remote_model("fast"), "deepseek-v4-pro")
            self.assertEqual(gateway.remote_model("deep"), "deepseek-v4-pro")
            self.assertEqual(
                gateway.remote_chat_url(),
                "https://api.deepseek.com/chat/completions",
            )

    def test_remote_model_names_can_be_overridden_for_private_gpu_service(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_REMOTE_URL": "http://192.168.1.20:8000/v1",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST": "qwen-fast",
                "PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP": "qwen-deep",
            },
            clear=False,
        ):
            gateway = ComputeGateway(_paths(Path(temp)))
            self.assertEqual(gateway.remote_model("fast"), "qwen-deep")
            self.assertEqual(gateway.remote_model("deep"), "qwen-deep")
            self.assertEqual(
                gateway.remote_chat_url(),
                "http://192.168.1.20:8000/v1/chat/completions",
            )
            self.assertFalse(gateway.remote_is_public())

    def test_local_llm_can_route_to_explicit_remote_without_local_model(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_REMOTE_URL": "http://127.0.0.1:8000/v1",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
            },
            clear=False,
        ):
            llm = LocalLLM(_paths(Path(temp)))
            self.assertEqual(llm.backend("fast"), "remote_server")
            self.assertTrue(llm.available("fast"))
            self.assertEqual(llm.active_model_name("fast"), "local-model")

    def test_deepspeed_request_falls_back_to_cuda_when_package_is_missing(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PHOENIX_KNOWLEDGE_ACCELERATOR": "deepspeed"},
            clear=False,
        ):
            gateway = ComputeGateway(_paths(Path(temp)))
            with patch.object(gateway, "_cuda_info", return_value=(True, 1, ("Test GPU",), (6.0,))), patch.object(
                gateway, "deepspeed_available", return_value=False
            ):
                status = gateway.status()
            self.assertEqual(status.effective_mode, "cuda")
            self.assertIn("回退普通CUDA", status.warning)

    def test_saved_settings_never_store_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            gateway = ComputeGateway(paths)
            with patch.dict(
                os.environ,
                {"PHOENIX_KNOWLEDGE_REMOTE_API_KEY": "secret-value"},
                clear=False,
            ):
                gateway.save_settings(
                    mode="remote",
                    remote_url="http://127.0.0.1:9000/v1",
                    remote_model_fast="fast-model",
                    remote_model_deep="deep-model",
                )
            text = gateway.config_path.read_text(encoding="utf-8")
            self.assertNotIn("secret-value", text)
            self.assertIn("fast-model", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
