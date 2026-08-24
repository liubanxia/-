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


def _legacy_url_only_configuration(gateway) -> bool:
    """Return True only for pre-Provider-Hub URL-only configurations."""

    if os.environ.get("PHOENIX_KNOWLEDGE_PROVIDER", "").strip():
        return False
    if os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST", "").strip():
        return False
    if os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP", "").strip():
        return False

    settings = getattr(gateway, "_settings", None)
    if str(getattr(settings, "remote_model_fast", "") or "").strip():
        return False
    if str(getattr(settings, "remote_model_deep", "") or "").strip():
        return False

    hub = getattr(gateway, "provider_hub", None)
    hub_path = getattr(hub, "path", None)
    if hub_path is not None:
        try:
            if hub_path.is_file():
                return False
        except OSError:
            return False
    return True


def install() -> None:
    """Keep old --gpu-url/compute_gateway.json settings protocol-neutral.

    Before Provider Hub existed, Phoenix stored only a remote URL. A custom
    OpenAI-compatible URL must not inherit DeepSeek-specific request fields just
    because DeepSeek is the default provider preset. Truly legacy private or
    loopback URL-only endpoints retain the historical ``local-model`` default.
    Once Provider Hub or per-profile model configuration is present, Smart1 and
    Smart2 remain independent and an intentionally blank profile stays
    unavailable.
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

    original_remote_model = ComputeGateway.remote_model

    def remote_model(self, profile=None) -> str:
        configured = str(original_remote_model(self, profile) or "").strip()
        if configured:
            return configured
        if (
            self.provider_id() == "custom_openai"
            and self.remote_url()
            and not self.remote_is_public()
            and _legacy_url_only_configuration(self)
        ):
            return "local-model"
        return ""

    ComputeGateway.provider_id = provider_id
    ComputeGateway.remote_model = remote_model
