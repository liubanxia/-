from __future__ import annotations

import os
import urllib.parse

from .provider_hub import DEFAULT_PROVIDER, PROVIDER_MAP

_INSTALLED = False


def _provider_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    host = (urllib.parse.urlparse(raw).hostname or "").lower()
    if not host:
        return ""
    for provider_id, spec in PROVIDER_MAP.items():
        if provider_id == "custom_openai" or not spec.base_url:
            continue
        spec_host = (urllib.parse.urlparse(spec.base_url).hostname or "").lower()
        if host == spec_host:
            return provider_id
    return "custom_openai"


def install() -> None:
    """Keep old --gpu-url/compute_gateway.json settings protocol-neutral.

    Before Provider Hub existed, Phoenix stored only a remote URL. A custom
    OpenAI-compatible URL must not inherit DeepSeek-specific request fields just
    because DeepSeek is the default provider preset.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .compute_gateway import ComputeGateway

    def provider_id(self) -> str:
        explicit = os.environ.get(
            "PHOENIX_KNOWLEDGE_PROVIDER",
            "",
        ).strip().lower()
        if explicit in PROVIDER_MAP:
            return explicit

        env_url = os.environ.get(
            "PHOENIX_KNOWLEDGE_REMOTE_URL",
            "",
        ).strip()
        if env_url:
            return _provider_from_url(env_url) or DEFAULT_PROVIDER

        hub = getattr(self, "provider_hub", None)
        if hub is not None and getattr(hub, "path", None) is not None:
            try:
                if hub.path.is_file() and hub.selected in PROVIDER_MAP:
                    return hub.selected
            except OSError:
                pass

        legacy_url = str(
            getattr(getattr(self, "_settings", None), "remote_url", "")
            or ""
        ).strip()
        if legacy_url:
            return _provider_from_url(legacy_url) or DEFAULT_PROVIDER
        return DEFAULT_PROVIDER

    ComputeGateway.provider_id = provider_id
