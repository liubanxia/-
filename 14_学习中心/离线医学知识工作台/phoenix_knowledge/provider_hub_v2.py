from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import replace

from . import provider_hub

_INSTALLED = False


def _responses_url(base_url: str) -> str:
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        return ""
    path = urllib.parse.urlparse(raw).path.rstrip("/")
    if path.endswith("/responses"):
        return raw
    if path.endswith("/v1"):
        return raw + "/responses"
    return raw + "/v1/responses"


def _extract_responses_text(data: dict) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    for output in data.get("output") or []:
        if not isinstance(output, dict):
            continue
        if output.get("type") == "message":
            for content in output.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    text = str(content.get("text") or "").strip()
                    if text:
                        parts.append(text)
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(f"Responses API 返回中没有可用文本: {data}")
    return text


def _install_current_presets() -> None:
    current = []
    for spec in provider_hub.PROVIDERS:
        if spec.id == "openai":
            spec = replace(
                spec,
                protocol="openai_responses",
                fast_model="gpt-5.6-luna",
                deep_model="gpt-5.6-sol",
                note=(
                    "OpenAI 官方 Responses API。智能1默认 GPT-5.6 Luna，"
                    "智能2默认 GPT-5.6 Sol；高级设置可覆盖模型名。"
                ),
            )
        current.append(spec)

    if not any(item.id == "hunyuan" for item in current):
        current.append(
            provider_hub.ProviderSpec(
                "hunyuan",
                "腾讯混元",
                "openai",
                "https://api.hunyuan.cloud.tencent.com/v1",
                "hunyuan-turbos-latest",
                "hunyuan-turbos-latest",
                "https://console.cloud.tencent.com/hunyuan",
                "HUNYUAN_API_KEY",
                "腾讯混元 OpenAI 兼容端口；模型迁移较快，"
                "高级设置可随平台更新直接替换模型 ID。",
            )
        )

    provider_hub.PROVIDERS = tuple(current)
    provider_hub.PROVIDER_MAP.clear()
    provider_hub.PROVIDER_MAP.update(
        {item.id: item for item in provider_hub.PROVIDERS}
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _install_current_presets()

    from .compute_gateway import ComputeGateway
    from .llm_safe import LocalLLM

    old_remote_generate = LocalLLM._remote_generate
    old_available = LocalLLM.available

    def remote_chat_url(self) -> str:
        protocol = self.current_provider().protocol
        if protocol == "openai_responses":
            return _responses_url(self.remote_url())
        return provider_hub._chat_url(self.remote_url(), protocol)

    ComputeGateway.remote_chat_url = remote_chat_url

    def available(self, profile=None):
        if not old_available(self, profile):
            return False
        if self.compute.requested_mode() != "remote":
            return True
        normalized = self._normalize_profile(profile)
        return bool(self.compute.remote_model(normalized).strip())

    LocalLLM.available = available

    def _remote_generate(self, prompt, max_new_tokens, profile=None):
        spec = self.compute.current_provider()
        if spec.protocol != "openai_responses":
            return old_remote_generate(
                self,
                prompt,
                max_new_tokens,
                profile,
            )

        status = self.compute.status()
        if status.effective_mode != "remote":
            raise RuntimeError(
                status.warning or "OpenAI 模型未授权或未配置"
            )
        key = self.compute.remote_api_key()
        if not key:
            raise RuntimeError("OpenAI 需要 API Key")
        normalized_profile = self._normalize_profile(profile)
        model = self.compute.remote_model(normalized_profile)
        if not model:
            raise RuntimeError("OpenAI 当前智能档位未配置模型 ID")
        url = self.compute.remote_chat_url()
        payload = {
            "model": model,
            "input": prompt,
            "max_output_tokens": int(max_new_tokens),
        }
        if normalized_profile == "translation":
            payload["reasoning"] = {"effort": "none"}
        request = urllib.request.Request(
            url,
            data=json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        timeout = self._timeout(
            "PHOENIX_KNOWLEDGE_REMOTE_TIMEOUT",
            180,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI 调用失败（{timeout}s）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return _extract_responses_text(data)

    LocalLLM._remote_generate = _remote_generate
