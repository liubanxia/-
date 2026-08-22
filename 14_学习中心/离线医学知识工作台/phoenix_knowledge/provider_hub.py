from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    protocol: str
    base_url: str
    fast_model: str
    deep_model: str
    console_url: str
    api_key_env: str
    note: str = ""


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        "deepseek",
        "DeepSeek",
        "openai",
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "https://platform.deepseek.com/",
        "DEEPSEEK_API_KEY",
        "官方 OpenAI 兼容接口；智能1=V4 Flash，智能2=V4 Pro。",
    ),
    ProviderSpec(
        "openai",
        "OpenAI",
        "openai",
        "https://api.openai.com/v1",
        "gpt-5-mini",
        "gpt-5.1",
        "https://platform.openai.com/",
        "OPENAI_API_KEY",
        "OpenAI 官方 API。模型名可在高级设置中覆盖。",
    ),
    ProviderSpec(
        "qwen",
        "阿里通义 Qwen",
        "openai",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3.7-plus",
        "qwen3.8-max",
        "https://bailian.console.aliyun.com/",
        "DASHSCOPE_API_KEY",
        "使用 DashScope OpenAI 兼容端口；企业可改为工作空间专属域名。",
    ),
    ProviderSpec(
        "zhipu",
        "智谱 GLM",
        "openai",
        "https://open.bigmodel.cn/api/paas/v4",
        "glm-4.7",
        "glm-5.2",
        "https://open.bigmodel.cn/",
        "ZHIPU_API_KEY",
        "智能1使用日常模型，智能2使用旗舰长程模型。",
    ),
    ProviderSpec(
        "kimi",
        "Kimi / Moonshot",
        "openai",
        "https://api.moonshot.cn/v1",
        "kimi-k2.6",
        "kimi-k2.6",
        "https://platform.moonshot.cn/",
        "MOONSHOT_API_KEY",
        "Kimi K2.6 同时支持普通与推理任务。",
    ),
    ProviderSpec(
        "gemini",
        "Google Gemini",
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.7-flash",
        "gemini-3.7-flash",
        "https://aistudio.google.com/",
        "GEMINI_API_KEY",
        "使用 Google 官方 OpenAI 兼容接口。",
    ),
    ProviderSpec(
        "siliconflow",
        "SiliconFlow 硅基流动",
        "openai",
        "https://api.siliconflow.cn/v1",
        "Qwen/Qwen3.5-9B",
        "Pro/zai-org/GLM-5",
        "https://cloud.siliconflow.cn/",
        "SILICONFLOW_API_KEY",
        "聚合多种开源模型；高级设置可直接填平台模型 ID。",
    ),
    ProviderSpec(
        "openrouter",
        "OpenRouter",
        "openai",
        "https://openrouter.ai/api/v1",
        "openrouter/auto",
        "openrouter/auto",
        "https://openrouter.ai/settings/keys",
        "OPENROUTER_API_KEY",
        "一个 API 接入多家模型；默认由 OpenRouter 自动选模与供应商回退。",
    ),
    ProviderSpec(
        "anthropic",
        "Anthropic Claude",
        "anthropic",
        "https://api.anthropic.com",
        "claude-sonnet-4-5",
        "claude-opus-4-6",
        "https://console.anthropic.com/",
        "ANTHROPIC_API_KEY",
        "使用 Anthropic Messages API；模型名可在高级设置中覆盖。",
    ),
    ProviderSpec(
        "custom_openai",
        "自定义 OpenAI 兼容端口",
        "openai",
        "",
        "",
        "",
        "",
        "PHOENIX_CUSTOM_API_KEY",
        "适用于局域网 vLLM/SGLang、火山方舟及其他 OpenAI 兼容平台。",
    ),
)

PROVIDER_MAP = {item.id: item for item in PROVIDERS}
DEFAULT_PROVIDER = "deepseek"
_INSTALLED = False


def provider_spec(provider_id: str | None) -> ProviderSpec:
    return PROVIDER_MAP.get(str(provider_id or "").strip().lower(), PROVIDER_MAP[DEFAULT_PROVIDER])


def provider_choices() -> tuple[ProviderSpec, ...]:
    return PROVIDERS


class ProviderState:
    def __init__(self, runtime_root: Path):
        self.path = Path(runtime_root) / "provider_hub.json"
        self.selected = DEFAULT_PROVIDER
        self.configs: dict[str, dict[str, str]] = {}
        self.reload()

    def reload(self) -> None:
        self.selected = DEFAULT_PROVIDER
        self.configs = {}
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            selected = str(payload.get("selected_provider") or DEFAULT_PROVIDER).strip().lower()
            if selected in PROVIDER_MAP:
                self.selected = selected
            configs = payload.get("providers") or {}
            if isinstance(configs, dict):
                self.configs = {
                    str(key): dict(value)
                    for key, value in configs.items()
                    if str(key) in PROVIDER_MAP and isinstance(value, dict)
                }
        except Exception:
            return

    def config(self, provider_id: str) -> dict[str, str]:
        return dict(self.configs.get(provider_id, {}))

    def save(
        self,
        provider_id: str,
        *,
        base_url: str = "",
        fast_model: str = "",
        deep_model: str = "",
    ) -> None:
        provider_id = str(provider_id).strip().lower()
        if provider_id not in PROVIDER_MAP:
            raise ValueError(f"未知模型平台: {provider_id}")
        self.selected = provider_id
        self.configs[provider_id] = {
            "base_url": str(base_url or "").strip(),
            "fast_model": str(fast_model or "").strip(),
            "deep_model": str(deep_model or "").strip(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(
                {
                    "selected_provider": self.selected,
                    "providers": self.configs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp.replace(self.path)


def _chat_url(base_url: str, protocol: str) -> str:
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        return ""
    path = urllib.parse.urlparse(raw).path.rstrip("/")
    if protocol == "anthropic":
        if path.endswith("/v1/messages"):
            return raw
        if path.endswith("/v1"):
            return raw + "/messages"
        return raw + "/v1/messages"
    if path.endswith("/chat/completions"):
        return raw
    if path.endswith("/v1") or path.endswith("/compatible-mode/v1") or path.endswith("/api/paas/v4"):
        return raw + "/chat/completions"
    return raw + "/chat/completions"


def _extract_openai_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"API响应缺少 choices: {data}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return text
    raise RuntimeError(f"API响应没有可用文本: {data}")


def _extract_anthropic_text(data: dict) -> str:
    parts = []
    for item in data.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise RuntimeError(f"Anthropic响应没有可用文本: {data}")
    return text


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .compute_gateway import ComputeGateway
    from .llm_safe import LocalLLM

    original_init = ComputeGateway.__init__
    original_reload = ComputeGateway.reload
    original_status = ComputeGateway.status

    def __init__(self, paths):
        original_init(self, paths)
        self.provider_hub = ProviderState(paths.runtime_root)

    def reload(self):
        original_reload(self)
        if hasattr(self, "provider_hub"):
            self.provider_hub.reload()

    def provider_id(self) -> str:
        env = os.environ.get("PHOENIX_KNOWLEDGE_PROVIDER", "").strip().lower()
        if env in PROVIDER_MAP:
            return env
        return getattr(self, "provider_hub", ProviderState(self.paths.runtime_root)).selected

    def current_provider(self) -> ProviderSpec:
        return provider_spec(self.provider_id())

    def provider_config(self, provider_id: str | None = None) -> dict[str, str]:
        pid = provider_id or self.provider_id()
        return self.provider_hub.config(pid)

    def select_provider(
        self,
        provider_id: str,
        *,
        base_url: str = "",
        fast_model: str = "",
        deep_model: str = "",
    ):
        spec = provider_spec(provider_id)
        url = str(base_url or spec.base_url).strip()
        fast = str(fast_model or spec.fast_model).strip()
        deep = str(deep_model or spec.deep_model).strip()
        self.provider_hub.save(
            spec.id,
            base_url=url,
            fast_model=fast,
            deep_model=deep,
        )
        os.environ["PHOENIX_KNOWLEDGE_PROVIDER"] = spec.id
        return self.save_settings(
            mode="remote",
            remote_url=url,
            remote_model_fast=fast,
            remote_model_deep=deep,
        )

    def remote_url(self) -> str:
        env = os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_URL", "").strip()
        if env:
            return env
        spec = self.current_provider()
        config = self.provider_hub.config(spec.id)
        return str(config.get("base_url") or spec.base_url or self._settings.remote_url).strip()

    def remote_api_key(self) -> str:
        generic = os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", "").strip()
        if generic:
            return generic
        spec = self.current_provider()
        return (
            os.environ.get(f"PHOENIX_PROVIDER_{spec.id.upper()}_API_KEY", "").strip()
            or os.environ.get(spec.api_key_env, "").strip()
        )

    def set_provider_api_key(self, provider_id: str, key: str) -> None:
        spec = provider_spec(provider_id)
        name = f"PHOENIX_PROVIDER_{spec.id.upper()}_API_KEY"
        if str(key or "").strip():
            os.environ[name] = str(key).strip()
        else:
            os.environ.pop(name, None)

    def remote_model(self, profile=None) -> str:
        normalized = str(profile or "fast").strip().lower()
        deep = normalized in {
            "deep",
            "4b",
            "deep4b",
            "quality",
            "max",
            "smart2",
            "translation",
            "medical_translation",
            "translate",
        }
        env_name = "PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP" if deep else "PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST"
        env = os.environ.get(env_name, "").strip()
        if env:
            return env
        spec = self.current_provider()
        config = self.provider_hub.config(spec.id)
        configured = str(config.get("deep_model" if deep else "fast_model") or "").strip()
        return configured or (spec.deep_model if deep else spec.fast_model)

    def provider_protocol(self) -> str:
        return self.current_provider().protocol

    def remote_chat_url(self) -> str:
        return _chat_url(self.remote_url(), self.provider_protocol())

    def is_deepseek_remote(self) -> bool:
        return self.provider_id() == "deepseek"

    def provider_label(self) -> str:
        return self.current_provider().label

    def status(self):
        result = original_status(self)
        if result.requested_mode != "remote":
            return result
        warning = result.warning
        effective = result.effective_mode
        if effective == "remote":
            if self.remote_is_public() and not self.remote_api_key():
                effective = "cpu"
                warning = f"{self.provider_label()} API Key 未配置"
            elif not self.remote_model("fast"):
                effective = "cpu"
                warning = f"{self.provider_label()} 快速模型 ID 未配置"
        return replace(result, effective_mode=effective, warning=warning)

    ComputeGateway.__init__ = __init__
    ComputeGateway.reload = reload
    ComputeGateway.provider_id = provider_id
    ComputeGateway.current_provider = current_provider
    ComputeGateway.provider_config = provider_config
    ComputeGateway.select_provider = select_provider
    ComputeGateway.remote_url = remote_url
    ComputeGateway.remote_api_key = remote_api_key
    ComputeGateway.set_provider_api_key = set_provider_api_key
    ComputeGateway.remote_model = remote_model
    ComputeGateway.provider_protocol = provider_protocol
    ComputeGateway.remote_chat_url = remote_chat_url
    ComputeGateway.is_deepseek_remote = is_deepseek_remote
    ComputeGateway.provider_label = provider_label
    ComputeGateway.status = status

    def _remote_generate(self, prompt, max_new_tokens, profile=None):
        status = self.compute.status()
        if status.effective_mode != "remote":
            raise RuntimeError(status.warning or "云端模型未授权或未配置")

        spec = self.compute.current_provider()
        model = self.compute.remote_model(self._normalize_profile(profile))
        url = self.compute.remote_chat_url()
        key = self.compute.remote_api_key()
        if not url:
            raise RuntimeError(f"{spec.label} API 地址为空")
        if self.compute.remote_is_public() and not key:
            raise RuntimeError(f"{spec.label} 需要 API Key")

        level = self._normalize_profile(profile)
        if spec.protocol == "anthropic":
            payload = {
                "model": model,
                "max_tokens": int(max_new_tokens),
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            if key:
                headers["x-api-key"] = key
        else:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": int(max_new_tokens),
                "stream": False,
            }
            if spec.id in {"deepseek", "zhipu"}:
                payload["thinking"] = {"type": "enabled" if level == "deep" else "disabled"}
            elif spec.id in {"qwen", "siliconflow"}:
                payload["enable_thinking"] = level == "deep"
            elif spec.id == "gemini":
                payload["reasoning_effort"] = "high" if level == "deep" else "low"
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            if spec.id == "openrouter":
                headers["X-Title"] = "Phoenix Medical Knowledge Workbench"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = self._timeout("PHOENIX_KNOWLEDGE_REMOTE_TIMEOUT", 180)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"{spec.label} 调用失败（{timeout}s）：{type(exc).__name__}: {exc}"
            ) from exc

        if spec.protocol == "anthropic":
            return _extract_anthropic_text(data)
        return _extract_openai_text(data)

    LocalLLM._remote_generate = _remote_generate
