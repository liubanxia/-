from __future__ import annotations

import os


_INSTALLED = False
_API_FLAG = "PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _api_fallback_allowed() -> bool:
    # Formal translation is offline/local-first by default. Remote API is only
    # an explicit emergency fallback, never a prerequisite for normal use.
    return _flag(_API_FLAG, default=False)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import hybrid_translation_policy as hybrid

    original_smart_available = hybrid._smart_available

    def translation_smart_available(engine) -> bool:
        if not _api_fallback_allowed():
            return False
        try:
            return bool(original_smart_available(engine))
        except Exception:
            return False

    # Every API branch in the retained local cascade consults this function.
    # Keeping it false by default removes mandatory API use and prevents token
    # burn from per-segment retries while preserving an explicit opt-in escape
    # hatch for exceptional cases.
    hybrid._smart_available = translation_smart_available

    # Acronym disambiguation follows the same policy. Deterministic local
    # terminology seeds/inventories still work with zero API calls. A local LLM
    # may still be used; only a remote backend is blocked unless explicitly
    # enabled through PHOENIX_TRANSLATION_ALLOW_API_FALLBACK=1.
    try:
        from .medical_acronyms import MedicalAcronymResolver

        original_llm_available = MedicalAcronymResolver._llm_available

        def acronym_llm_available(self) -> bool:
            try:
                if not original_llm_available(self):
                    return False
                llm = getattr(self, "llm", None)
                if llm is None:
                    return False
                backend = str(llm.backend("translation") or "")
                if backend == "remote_server" and not _api_fallback_allowed():
                    return False
                return True
            except Exception:
                return False

        MedicalAcronymResolver._llm_available = acronym_llm_available
    except Exception:
        pass

    # Production translation route:
    # HY-MT 1.8B local -> local Qwen model3 only when HY-MT misses the quality
    # gate -> optional API only when explicitly enabled. Full-page model3/API
    # review remains disabled, so the earlier minute-scale review regression
    # does not return.
    from .hymt_cascade_policy import install as install_hymt
    from .translation_cascade_v2 import install as install_local_cascade

    install_hymt()
    install_local_cascade()

    # Old warning-free checkpoints may have been produced by the previous
    # Smart2-qwen route while the compute gateway was in remote/API mode. Their
    # backend label does not distinguish local-vs-remote execution, so when API
    # fallback is disabled we invalidate those legacy qwen35 unit caches once.
    # Newly generated HY-MT/model3 checkpoints are retained normally.
    try:
        from .office_translation import OfficeDocumentTranslator

        previous_load_completed = OfficeDocumentTranslator._load_completed_unit

        def load_completed_unit(
            path,
            unit,
            *,
            source_sha256: str,
            target_language: str,
            glossary_sha256: str,
        ):
            completed = previous_load_completed(
                path,
                unit,
                source_sha256=source_sha256,
                target_language=target_language,
                glossary_sha256=glossary_sha256,
            )
            if completed is None or _api_fallback_allowed():
                return completed
            _translated, _warnings, audits = completed
            for row in audits:
                if not isinstance(row, dict):
                    continue
                backend = str(row.get("backend", "") or "")
                if backend.startswith("qwen35_medical_translation"):
                    return None
            return completed

        OfficeDocumentTranslator._load_completed_unit = staticmethod(load_completed_unit)
    except Exception:
        pass

    print(
        "[Phoenix][翻译] 本地优先正式模式：HY-MT→本地模型3；API默认关闭，"
        "仅设置 PHOENIX_TRANSLATION_ALLOW_API_FALLBACK=1 时作为最终兜底。",
        flush=True,
    )
