from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.compute_gateway import ComputeGateway
from phoenix_knowledge.compute_gui import _local_or_private_host
from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.llm_safe import LocalLLM
from phoenix_knowledge.provider_hub import (
    PROVIDER_MAP,
    _chat_url,
    provider_choices,
)
from phoenix_knowledge.provider_hub_compat import _provider_from_url
from phoenix_knowledge.provider_hub_v2 import (
    _extract_responses_text,
    _responses_url,
)


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


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class ProviderHubTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for name in list(os.environ):
            if name.startswith("PHOENIX_PROVIDER_") or name.startswith(
                "PHOENIX_KNOWLEDGE_REMOTE_"
            ):
                os.environ.pop(name, None)
        for name in (
            "PHOENIX_KNOWLEDGE_PROVIDER",
            "PHOENIX_KNOWLEDGE_ACCELERATOR",
            "PHOENIX_KNOWLEDGE_ALLOW_REMOTE",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "DASHSCOPE_API_KEY",
            "ZHIPU_API_KEY",
            "MOONSHOT_API_KEY",
            "GEMINI_API_KEY",
            "SILICONFLOW_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "HUNYUAN_API_KEY",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        self.env.stop()

    def test_major_provider_choices_are_present(self):
        ids = {item.id for item in provider_choices()}
        self.assertTrue(
            {
                "deepseek",
                "openai",
                "qwen",
                "zhipu",
                "kimi",
                "gemini",
                "siliconflow",
                "openrouter",
                "anthropic",
                "hunyuan",
                "custom_openai",
            }.issubset(ids)
        )
        self.assertEqual(
            PROVIDER_MAP["deepseek"].fast_model,
            "deepseek-v4-flash",
        )
        self.assertEqual(
            PROVIDER_MAP["kimi"].fast_model,
            "kimi-k2.6",
        )
        self.assertEqual(
            PROVIDER_MAP["openai"].protocol,
            "openai_responses",
        )
        self.assertEqual(
            PROVIDER_MAP["openai"].fast_model,
            "gpt-5.6-luna",
        )
        self.assertEqual(
            PROVIDER_MAP["openai"].deep_model,
            "gpt-5.6-sol",
        )
        self.assertEqual(
            PROVIDER_MAP["hunyuan"].base_url,
            "https://api.hunyuan.cloud.tencent.com/v1",
        )

    def test_protocol_urls_are_built_without_double_v1(self):
        self.assertEqual(
            _chat_url("https://api.moonshot.cn/v1", "openai"),
            "https://api.moonshot.cn/v1/chat/completions",
        )
        self.assertEqual(
            _chat_url("https://api.anthropic.com", "anthropic"),
            "https://api.anthropic.com/v1/messages",
        )
        self.assertEqual(
            _chat_url(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "openai",
            ),
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        self.assertEqual(
            _responses_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses",
        )

    def test_responses_text_extracts_message_output(self):
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "医学回答"}
                    ],
                }
            ]
        }
        self.assertEqual(
            _extract_responses_text(payload),
            "医学回答",
        )

    def test_old_custom_gpu_url_is_not_misclassified_as_deepseek(self):
        self.assertEqual(
            _provider_from_url("http://127.0.0.1:8000/v1"),
            "custom_openai",
        )
        self.assertEqual(
            _provider_from_url("https://api.deepseek.com"),
            "deepseek",
        )
        self.assertEqual(
            _provider_from_url("https://api.moonshot.cn/v1"),
            "kimi",
        )
        self.assertEqual(
            _provider_from_url(
                "https://api.hunyuan.cloud.tencent.com/v1"
            ),
            "hunyuan",
        )

    def test_provider_settings_persist_but_api_key_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            gateway = ComputeGateway(paths)
            gateway.set_provider_api_key("deepseek", "secret-key")
            gateway.select_provider(
                "deepseek",
                base_url="https://api.deepseek.com",
                fast_model="deepseek-v4-flash",
                deep_model="deepseek-v4-pro",
            )
            payload = json.loads(
                (paths.runtime_root / "provider_hub.json").read_text(
                    encoding="utf-8"
                )
            )
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("secret-key", serialized)
            self.assertEqual(payload["selected_provider"], "deepseek")
            self.assertEqual(
                payload["providers"]["deepseek"]["deep_model"],
                "deepseek-v4-pro",
            )

    def test_public_provider_is_not_ready_without_key(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "openai",
            },
            clear=False,
        ):
            gateway = ComputeGateway(_paths(Path(temp)))
            gateway.select_provider("openai")
            os.environ.pop("PHOENIX_PROVIDER_OPENAI_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            status = gateway.status()
            self.assertNotEqual(status.effective_mode, "remote")
            self.assertIn("API Key", status.warning)

    def test_private_lan_detection_matches_backend_policy(self):
        self.assertTrue(_local_or_private_host("127.0.0.1"))
        self.assertTrue(_local_or_private_host("192.168.1.20"))
        self.assertTrue(_local_or_private_host("10.0.0.8"))
        self.assertFalse(_local_or_private_host("8.8.8.8"))

    def test_loopback_custom_openai_can_run_without_key(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "custom_openai",
            },
            clear=False,
        ):
            gateway = ComputeGateway(_paths(Path(temp)))
            gateway.select_provider(
                "custom_openai",
                base_url="http://127.0.0.1:8000/v1",
                fast_model="local-fast",
                deep_model="local-deep",
            )
            self.assertEqual(gateway.status().effective_mode, "remote")
            self.assertEqual(
                gateway.remote_chat_url(),
                "http://127.0.0.1:8000/v1/chat/completions",
            )

    def test_openai_uses_responses_api(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "openai",
            },
            clear=False,
        ):
            paths = _paths(Path(temp))
            llm = LocalLLM(paths)
            llm.compute.set_provider_api_key("openai", "openai-secret")
            llm.compute.select_provider("openai")
            captured = {}

            def fake_urlopen(request, timeout=None):
                captured["request"] = request
                return _FakeResponse(
                    {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "证据归纳",
                                    }
                                ],
                            }
                        ]
                    }
                )

            with patch(
                "phoenix_knowledge.provider_hub_v2.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                text = llm._remote_generate("测试", 64, "fast")

            self.assertEqual(text, "证据归纳")
            request = captured["request"]
            self.assertEqual(
                request.full_url,
                "https://api.openai.com/v1/responses",
            )
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "gpt-5.6-sol")
            self.assertEqual(body["input"], "测试")
            self.assertEqual(body["max_output_tokens"], 64)
            self.assertNotIn("temperature", body)

    def test_openai_translation_uses_quality_model_without_reasoning(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "openai",
            },
            clear=False,
        ):
            paths = _paths(Path(temp))
            llm = LocalLLM(paths)
            llm.compute.set_provider_api_key("openai", "openai-secret")
            llm.compute.select_provider("openai")
            captured = {}

            def fake_urlopen(request, timeout=None):
                captured["request"] = request
                return _FakeResponse(
                    {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "医学译文",
                                    }
                                ],
                            }
                        ]
                    }
                )

            with patch(
                "phoenix_knowledge.provider_hub_v2.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                text = llm._remote_generate(
                    "翻译测试",
                    512,
                    "translation",
                )

            self.assertEqual(text, "医学译文")
            body = json.loads(
                captured["request"].data.decode("utf-8")
            )
            self.assertEqual(body["model"], "gpt-5.6-sol")
            self.assertEqual(body["max_output_tokens"], 512)
            self.assertEqual(
                body["reasoning"],
                {"effort": "none"},
            )

    def test_deepseek_translation_uses_quality_model_with_thinking_disabled(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "deepseek",
            },
            clear=False,
        ):
            llm = LocalLLM(_paths(Path(temp)))
            llm.compute.set_provider_api_key("deepseek", "deepseek-secret")
            llm.compute.select_provider("deepseek")
            captured = {}

            def fake_urlopen(request, timeout=None):
                captured["request"] = request
                return _FakeResponse(
                    {"choices": [{"message": {"content": "医学译文"}}]}
                )

            with patch(
                "phoenix_knowledge.provider_hub.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                text = llm._remote_generate(
                    "翻译测试",
                    512,
                    "translation",
                )

            self.assertEqual(text, "医学译文")
            body = json.loads(captured["request"].data.decode("utf-8"))
            self.assertEqual(body["model"], "deepseek-v4-pro")
            self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_anthropic_native_request_and_response(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "anthropic",
            },
            clear=False,
        ):
            paths = _paths(Path(temp))
            llm = LocalLLM(paths)
            llm.compute.set_provider_api_key("anthropic", "anthropic-secret")
            llm.compute.select_provider(
                "anthropic",
                base_url="https://api.anthropic.com",
                fast_model="claude-test-fast",
                deep_model="claude-test-deep",
            )
            captured = {}

            def fake_urlopen(request, timeout=None):
                captured["request"] = request
                captured["timeout"] = timeout
                return _FakeResponse(
                    {"content": [{"type": "text", "text": "医学回答"}]}
                )

            with patch(
                "phoenix_knowledge.provider_hub.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                text = llm._remote_generate("测试", 32, "fast")

            self.assertEqual(text, "医学回答")
            request = captured["request"]
            self.assertTrue(request.full_url.endswith("/v1/messages"))
            headers = {
                key.lower(): value
                for key, value in request.header_items()
            }
            self.assertEqual(
                headers.get("x-api-key"),
                "anthropic-secret",
            )
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "claude-test-deep")
            self.assertEqual(body["messages"][0]["content"], "测试")


if __name__ == "__main__":
    unittest.main(verbosity=2)
