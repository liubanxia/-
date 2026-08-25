from __future__ import annotations

"""Canonical production contract for formal medical translation readiness/routing.

This module closes two release bugs that could coexist in the same process:

* the compute dialog could truthfully report a ready remote provider while the
  PDF/Office preflight still used the old ``active_backends`` Smart2 inventory
  and rejected the job as not ready;
* the v3 quality-chain implementation existed, but production routing could
  still retain the older dual-route local cascade because the final invariant
  was not installed as the last translation contract.

The v4 contract makes one source of truth serve PDF, Office, the GUI status,
and the actual translation cascade:

    M1 (when available) -> M2 (when available) -> M3 (when available)
    -> selected remote provider only when the local chain cannot publish safely.

Selecting remote compute and granting the workbench's explicit session consent
is also sufficient consent for the translation fallback.  The historical
translation-specific environment flag is synchronized for compatibility rather
than being allowed to disagree with the compute dialog.
"""

import os
from dataclasses import dataclass

from .translation_models import MultiModelTranslationEngine, _normalize_smart_level


_INSTALLED = False
_API_FLAG = "PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"
_KNOWLEDGE_REMOTE_FLAG = "PHOENIX_KNOWLEDGE_ALLOW_REMOTE"


@dataclass(frozen=True)
class _StageInventoryBackend:
    """Readiness-only backend descriptor; never used to perform inference."""

    name: str


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _remote_api_ready(engine: MultiModelTranslationEngine) -> bool:
    """Return the same remote readiness truth used by the compute gateway.

    A provider shown as effective remote compute must not be rejected by a
    second, stale translation-only readiness test.  The workbench permission is
    the user-facing consent switch; the old translation flag is synchronized so
    legacy helpers observe the same decision.
    """

    llm = getattr(getattr(engine, "qwen", None), "llm", None)
    compute = getattr(llm, "compute", None)
    if llm is None or compute is None:
        return False

    try:
        if str(compute.requested_mode() or "").strip().lower() != "remote":
            return False
        status = compute.status()
        if str(getattr(status, "effective_mode", "") or "").strip().lower() != "remote":
            return False
        if not bool(getattr(status, "remote_allowed", False)):
            return False
        # ``remote_allowed`` is derived from the explicit session consent shown
        # in the model/compute dialog.  Keep the legacy flag in sync so older
        # ancillary code (acronym/cache helpers) cannot disagree with the route.
        os.environ[_KNOWLEDGE_REMOTE_FLAG] = "1"
        os.environ[_API_FLAG] = "1"

        model = str(compute.remote_model("translation") or "").strip()
        if not model:
            return False
        if bool(compute.remote_is_public()) and not str(compute.remote_api_key() or "").strip():
            return False
        # Provider Hub adds protocol/model-specific checks to LocalLLM.available.
        if not bool(llm.available("translation")):
            return False
        return True
    except Exception:
        return False


def chain_status(engine: MultiModelTranslationEngine) -> dict:
    """Cheap, non-loading readiness inventory used by GUI and preflight."""

    from . import hymt_cascade_policy as hymt
    from . import translation_cascade_v2 as cascade
    from . import translation_dual_route_release as dual

    m1_names: list[str] = []
    try:
        if bool(dual._fast_local_ready(engine)):
            m1_names.append("local-fast")
    except Exception:
        pass

    # Retained local seq2seq weights are valid model-1 inventory.  They are not
    # allowed to masquerade as a complete formal route; v4 merely reports that
    # the first stage exists and the later quality stages still decide release.
    for attr in ("nllb", "marian"):
        backend = getattr(engine, attr, None)
        if backend is None:
            continue
        try:
            ready = bool(engine._backend_available(backend))
        except Exception:
            ready = False
        if ready:
            name = str(getattr(backend, "name", attr) or attr)
            if name not in m1_names:
                m1_names.append(name)

    try:
        m2_ready = bool(hymt._model2_available(engine))
        m2_path = str(getattr(hymt._model2(engine), "model_path", ""))
    except Exception:
        m2_ready, m2_path = False, ""

    try:
        m3_ready = bool(cascade._model3_available(engine))
        m3_path = str(getattr(cascade._model3(engine), "model_path", ""))
    except Exception:
        m3_ready, m3_path = False, ""

    api_ready = _remote_api_ready(engine)
    return {
        "model1_ready": bool(m1_names),
        "model1_names": tuple(m1_names),
        "model2_ready": m2_ready,
        "model2_path": m2_path,
        "model3_ready": m3_ready,
        "model3_path": m3_path,
        "api_ready": api_ready,
        "formal_ready": bool(m1_names) or m2_ready or m3_ready or api_ready,
    }


def _provider_inventory_name(engine: MultiModelTranslationEngine) -> str:
    llm = getattr(getattr(engine, "qwen", None), "llm", None)
    compute = getattr(llm, "compute", None)
    if compute is None:
        return "Smart2/API"
    try:
        label = str(compute.provider_label() or "API").strip()
    except Exception:
        label = "API"
    try:
        model = str(compute.remote_model("translation") or "").strip()
    except Exception:
        model = ""
    return f"API:{label}/{model}" if model else f"API:{label}"


def formal_stage_names(engine: MultiModelTranslationEngine) -> tuple[str, ...]:
    status = chain_status(engine)
    names: list[str] = []
    if status["model1_ready"]:
        detail = ",".join(status["model1_names"])
        names.append(f"M1:{detail}" if detail else "M1")
    if status["model2_ready"]:
        names.append("M2:HY-MT1.5-1.8B")
    if status["model3_ready"]:
        names.append("M3:Qwen-local-medical")
    if status["api_ready"]:
        names.append(_provider_inventory_name(engine))
    return tuple(names)


def _inventory_backends(engine: MultiModelTranslationEngine) -> list[_StageInventoryBackend]:
    return [_StageInventoryBackend(name) for name in formal_stage_names(engine)]


def _report_inventory(engine: MultiModelTranslationEngine) -> None:
    if bool(getattr(engine, "_phoenix_chain_inventory_reported_v4", False)):
        return
    engine._phoenix_chain_inventory_reported_v4 = True
    status = chain_status(engine)
    m1 = (
        "READY[" + ",".join(status["model1_names"]) + "]"
        if status["model1_ready"]
        else "NOT READY"
    )
    print(
        "[Phoenix][翻译链v4] "
        f"M1={m1} -> "
        f"M2={'READY' if status['model2_ready'] else 'NOT READY'} -> "
        f"M3={'READY' if status['model3_ready'] else 'NOT READY'} -> "
        f"API={'READY' if status['api_ready'] else 'NOT READY'}",
        flush=True,
    )


def _report_route(engine: MultiModelTranslationEngine, result) -> None:
    count = int(getattr(engine, "_phoenix_chain_trace_count_v4", 0)) + 1
    engine._phoenix_chain_trace_count_v4 = count
    attempts = tuple(getattr(result, "attempts", ()) or ())
    m1 = m2 = m3 = api = False
    for item in attempts:
        name = str(getattr(item, "backend", "") or "").lower()
        if (
            name.startswith("model1_")
            or name.startswith("model1_draft:")
            or name.startswith("marian_en_zh")
            or name.startswith("nllb_600m_en_zh")
        ):
            m1 = True
        if name.startswith("hymt15_1p8b"):
            m2 = True
        if name.startswith("qwen_local_medical_model3"):
            m3 = True
        if name.startswith("qwen35_medical_translation"):
            api = True

    if count <= 10 or count % 25 == 0 or api:
        final_backend = str(getattr(result, "backend", "") or "unknown")
        score = float(getattr(getattr(result, "quality", None), "score", 0.0) or 0.0)
        print(
            f"[Phoenix][翻译链实况v4] #{count} "
            f"M1={'RUN' if m1 else 'SKIP'} -> "
            f"M2={'RUN' if m2 else 'SKIP'} -> "
            f"M3={'RUN' if m3 else 'SKIP'} -> "
            f"API={'RUN' if api else 'SKIP'} | "
            f"final={final_backend} | score={score:.2f}",
            flush=True,
        )


def install() -> None:
    """Install the one final production readiness/routing contract."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import hybrid_translation_policy as hybrid
    from . import translation_cascade_v2 as cascade
    from .translation_chain_enforcement_v3 import _run_quality_chain

    cls = MultiModelTranslationEngine

    # 1) One API truth.  The same selected provider/consent/key/model that makes
    # the compute dialog READY makes the final translation fallback READY.
    hybrid._smart_available = _remote_api_ready

    # 2) One local quality route.  M2/M3 are quality stages, not merely optional
    # labels; when their local weights exist they are attempted in sequence.
    cascade._run_local_cascade = _run_quality_chain

    previous_accept = cascade._local_draft_accepted

    def local_draft_accepted(local_draft, local_stage: str) -> bool:
        stage = str(local_stage or "")
        if stage in {"quality_final_model3", "quality_final_model3_source"}:
            return bool(local_draft.quality.ok) and float(local_draft.quality.score) >= cascade.MODEL3_ACCEPT_SCORE
        if stage == "quality_model3_unavailable":
            # If no final local reviewer exists, a structurally safe M1/M2
            # candidate may publish without forcing a paid API call.
            return bool(local_draft.quality.ok) and float(local_draft.quality.score) >= 0.62
        return previous_accept(local_draft, stage)

    cascade._local_draft_accepted = local_draft_accepted

    # 3) Replace the obsolete Smart2-only preflight inventory.  PDF and Office
    # both call engine.active_backends(..., 'smart2') before starting.  Return
    # readiness descriptors for every real stage so local M2/M3 or a ready API
    # can no longer be rejected before the cascade gets a chance to run.
    previous_active_backends = cls.active_backends

    def active_backends(self, target_language: str = "中文", smart_level: str = "smart1"):
        level = _normalize_smart_level(smart_level)
        if level != "smart2":
            return previous_active_backends(self, target_language, smart_level)
        return _inventory_backends(self)

    def formal_backend_names(self, target_language: str = "中文") -> list[str]:
        del target_language
        return list(formal_stage_names(self))

    cls.active_backends = active_backends
    cls.formal_backend_names = formal_backend_names

    # 4) Wrap the actual class entry point, not just a module-global function,
    # so telemetry proves what production executed.
    previous_translate = cls.translate

    def translate(self, source: str, target_language: str = "中文", *, smart_level: str = "smart1"):
        level = _normalize_smart_level(smart_level)
        if level == "smart2":
            _report_inventory(self)
        result = previous_translate(
            self,
            source,
            target_language,
            smart_level=level,
        )
        if level == "smart2":
            _report_route(self, result)
        return result

    translate._phoenix_translation_runtime_contract_v4 = True
    cls.translate = translate
    cls._phoenix_translation_runtime_contract_v4 = 4

    _INSTALLED = True
    print(
        "[Phoenix][翻译运行时] v4已启用：统一PDF/Office/GUI/API就绪判定；"
        "M1->M2->M3本地质量链优先，远程API仅在本地链不能安全发布时兜底。",
        flush=True,
    )
