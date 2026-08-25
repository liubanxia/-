from __future__ import annotations

"""Final production contract for the medical translation cascade.

The v3 contextual route had accidentally weakened the preceding quality-first
contract in two ways: model2 only ran after a model1 failure, and model3 could be
skipped entirely when model1/model2 produced no draft. This module restores the
intended production invariant:

    model1 (when available) -> model2 (when available) -> model3 (when available)
    -> Smart2/API only after the local chain cannot publish safely.

All models remain lazy-loaded. Readiness checks inspect local model inventory;
weights are loaded only when a translation actually reaches that stage.
"""

_INSTALLED = False
_FINAL_TAG = "|quality_final_v2"


def _model3_source_translate(backend, source: str, target_language: str) -> str:
    system = (
        "你是 Phoenix 本地医学翻译模型3。当前前两级没有产生可用中文初稿，请直接根据英文原文完成终审级翻译。"
        "必须完整保留疾病、解剖、影像学、病理、检查技术、药物、统计学术语，以及全部数字、单位、"
        "正负号、侧别、否定关系、分级、诊断确定性、医学缩写和图表编号。"
        "禁止总结、删减、扩写、解释、拒答或添加原文没有的医学知识。只输出最终译文。"
    )
    user = (
        f"目标语言：{target_language}\n\n"
        f"英文原文：\n{source}\n\n"
        "请直接输出完整、准确的医学译文。"
    )
    backend._load()
    prompt = backend._chat_prompt(system, user)
    return backend._generate_prompt(
        prompt,
        str(source or ""),
        mode_label="v3模型3源文直译",
        max_input_length=1792,
        max_output_tokens=768,
    )


def _run_quality_chain(
    engine,
    source: str,
    target_language: str,
    attempts: list,
    errors: list[str],
):
    from . import hymt_cascade_policy as hymt
    from . import translation_cascade_v2 as cascade
    from . import translation_dual_route_release as dual

    model1 = dual._model1(
        engine,
        source,
        target_language,
        attempts,
        errors,
    )

    # Model2 is a quality stage, not merely a model1 failure fallback. If its
    # local weights exist, every formal medical segment must pass through it.
    model2 = hymt._run_model2(
        engine,
        source,
        model1,
        target_language,
        attempts,
        errors,
    )

    base = (
        model2
        if model2 is not None and str(getattr(model2, "text", "") or "").strip()
        else model1
    )

    # An installed model3 must never be skipped because M1/M2 produced no draft.
    if not cascade._model3_available(engine):
        if base is None:
            return None, "quality_no_local_draft"
        return base, "quality_model3_unavailable"

    backend = cascade._model3(engine)
    try:
        if base is not None and str(getattr(base, "text", "") or "").strip():
            text = backend.refine(source, base.text, target_language)
            backend_name = backend.name + _FINAL_TAG
            stage = "quality_final_model3"
        else:
            text = _model3_source_translate(backend, source, target_language)
            backend_name = backend.name + ":source" + _FINAL_TAG
            stage = "quality_final_model3_source"

        final = hymt._quality_attempt(
            engine,
            backend_name,
            source,
            text,
            target_language,
        )
        attempts.append(final)
        return final, stage
    except Exception as exc:
        errors.append(f"Qwen-model3-final: {type(exc).__name__}: {exc}")
        if base is None:
            return None, "quality_model3_failed_no_draft"
        return base, "quality_model3_failed"


def _classify_attempts(result) -> tuple[bool, bool, bool, bool]:
    model1 = model2 = model3 = api = False
    for item in tuple(getattr(result, "attempts", ()) or ()):
        name = str(getattr(item, "backend", "") or "").lower()
        if (
            name.startswith("model1_")
            or name.startswith("model1_draft:")
            or name.startswith("marian_en_zh")
            or name.startswith("nllb_600m_en_zh")
        ):
            model1 = True
        if name.startswith("hymt15_1p8b"):
            model2 = True
        if name.startswith("qwen_local_medical_model3"):
            model3 = True
        if name.startswith("qwen35_medical_translation"):
            api = True
    return model1, model2, model3, api


def chain_status(engine) -> dict:
    """Return cheap readiness truth without loading model weights."""

    from . import hybrid_translation_policy as hybrid
    from . import hymt_cascade_policy as hymt
    from . import translation_cascade_v2 as cascade
    from . import translation_dual_route_release as dual

    m1_names: list[str] = []
    try:
        if dual._fast_local_ready(engine):
            m1_names.append("local-fast")
    except Exception:
        pass
    if not m1_names:
        for attr in ("nllb", "marian"):
            backend = getattr(engine, attr, None)
            if backend is None:
                continue
            try:
                if bool(engine._backend_available(backend)):
                    m1_names.append(str(getattr(backend, "name", attr)))
            except Exception:
                pass

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
    try:
        api_ready = bool(hybrid._smart_available(engine))
    except Exception:
        api_ready = False

    return {
        "model1_ready": bool(m1_names),
        "model1_names": tuple(m1_names),
        "model2_ready": m2_ready,
        "model2_path": m2_path,
        "model3_ready": m3_ready,
        "model3_path": m3_path,
        "api_ready": api_ready,
    }


def _report_inventory(engine) -> None:
    if bool(getattr(engine, "_phoenix_chain_inventory_reported_v3", False)):
        return
    engine._phoenix_chain_inventory_reported_v3 = True

    status = chain_status(engine)
    names = status["model1_names"]
    m1_text = "READY[" + ",".join(names) + "]" if names else "NOT READY"
    print(
        "[Phoenix][本地翻译链状态] "
        f"M1={m1_text} | "
        f"M2={'READY' if status['model2_ready'] else 'NOT READY'} | "
        f"M3={'READY' if status['model3_ready'] else 'NOT READY'} | "
        f"Smart2/API={'READY' if status['api_ready'] else 'NOT READY'}",
        flush=True,
    )
    if status["model2_path"]:
        print(f"[Phoenix][模型2路径] {status['model2_path']}", flush=True)
    if status["model3_path"]:
        print(f"[Phoenix][模型3路径] {status['model3_path']}", flush=True)


def _report_route(engine, result) -> None:
    count = int(getattr(engine, "_phoenix_chain_trace_count_v3", 0)) + 1
    engine._phoenix_chain_trace_count_v3 = count
    m1, m2, m3, api = _classify_attempts(result)
    final_backend = str(getattr(result, "backend", "") or "unknown")
    score = float(getattr(getattr(result, "quality", None), "score", 0.0) or 0.0)

    if count <= 10 or count % 25 == 0 or api:
        print(
            f"[Phoenix][翻译链实况] #{count} "
            f"M1={'RUN' if m1 else 'SKIP'} -> "
            f"M2={'RUN' if m2 else 'SKIP'} -> "
            f"M3={'RUN' if m3 else 'SKIP'} -> "
            f"API={'RUN' if api else 'SKIP'} | "
            f"final={final_backend} | score={score:.2f}",
            flush=True,
        )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import translation_cascade_v2 as cascade
    from .translation_models import _normalize_smart_level

    cascade._run_local_cascade = _run_quality_chain
    old_translate = cascade._translate

    def translate(
        engine,
        source: str,
        target_language: str = "中文",
        *,
        smart_level: str = "smart1",
    ):
        level = _normalize_smart_level(smart_level)
        if level == "smart2":
            _report_inventory(engine)
        result = old_translate(
            engine,
            source,
            target_language,
            smart_level=level,
        )
        if level == "smart2":
            _report_route(engine, result)
        return result

    translate._phoenix_chain_enforcement_v3 = True
    cascade._translate = translate
    _INSTALLED = True
    print(
        "[Phoenix][翻译链强制] v3已启用：M1可用则初译→M2可用则必经→M3可用则必经终审；"
        "前两级无稿时M3直接源文翻译；Smart2/API仅在本地链仍未通过时兜底。",
        flush=True,
    )
