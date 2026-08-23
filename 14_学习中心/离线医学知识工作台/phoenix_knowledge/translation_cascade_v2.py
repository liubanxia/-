from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from . import hybrid_translation_policy as hybrid
from . import hymt_cascade_policy as cascade
from .qwen_local_medical_backend import LocalQwenMedicalBackend
from .translation_models import (
    LEGACY_PREVIEW_BACKEND_NAMES,
    MultiModelTranslationEngine,
    TranslationAttempt,
    TranslationDecision,
    _normalize_smart_level,
)


MODEL3_ACCEPT_SCORE = 0.62
_API_FINAL_POLISH_REASON = (
    "这是本地翻译模型已经完成的中文医学译文。你的任务不是从头重译，也不是总结；"
    "请把现有译文作为工作底稿，严格对照英文原文进行最终医学精修。优先修正医学术语、"
    "漏译、误译、生硬语序和上下文表达，同时必须保持全部数字、单位、正负号、侧别、"
    "否定关系、分级、医学缩写、图表编号和诊断确定性。只输出精修后的完整译文。"
)


def _best_attempt(attempts: Iterable[TranslationAttempt]) -> TranslationAttempt | None:
    values = [item for item in attempts if str(getattr(item, "text", "") or "").strip()]
    if not values:
        return None
    return max(values, key=lambda item: float(item.quality.score))


def _model3_always() -> bool:
    raw = os.environ.get("PHOENIX_MODEL3_ALWAYS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _report_model3_fast_mode(engine: MultiModelTranslationEngine) -> None:
    if bool(getattr(engine, "_phoenix_model3_fast_mode_reported", False)):
        return
    print(
        "[Phoenix][模型3] 快速模式：仅当前两级本地译文未通过质量门槛时调用；"
        "设置 PHOENIX_MODEL3_ALWAYS=1 可强制每段都精修。",
        flush=True,
    )
    engine._phoenix_model3_fast_mode_reported = True


def _model3_path(engine: MultiModelTranslationEngine) -> Path:
    override = os.environ.get("PHOENIX_QWEN_MODEL3_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return (
        engine.paths.project_root
        / "04_AI模型"
        / "translation"
        / "model3"
        / "Qwen2.5-3B-Instruct"
    )


def _model3(engine: MultiModelTranslationEngine) -> LocalQwenMedicalBackend:
    backend = getattr(engine, "_phoenix_qwen_model3", None)
    expected = _model3_path(engine)
    if backend is None or Path(getattr(backend, "model_path", "")) != expected:
        backend = LocalQwenMedicalBackend(expected)
        engine._phoenix_qwen_model3 = backend
    return backend


def _model3_available(engine: MultiModelTranslationEngine) -> bool:
    try:
        return _model3(engine).available()
    except Exception:
        return False


def _run_model1(
    engine: MultiModelTranslationEngine,
    source: str,
    target_language: str,
    attempts: list[TranslationAttempt],
    errors: list[str],
) -> tuple[TranslationAttempt | None, bool]:
    """Return best model-1 draft and whether it passed the model-1 gate."""

    best: TranslationAttempt | None = None
    for backend in hybrid._local_backends(engine, target_language):
        try:
            attempt = hybrid._attempt(engine, backend, source, target_language)
            attempts.append(attempt)
            if best is None or attempt.quality.score > best.quality.score:
                best = attempt
        except Exception as exc:
            errors.append(
                f"model1-{getattr(backend, 'name', 'local')}: "
                f"{type(exc).__name__}: {exc}"
            )

    passed = bool(
        best is not None
        and best.quality.ok
        and float(best.quality.score) >= cascade.MODEL1_ACCEPT_SCORE
    )
    return best, passed


def _run_model3(
    engine: MultiModelTranslationEngine,
    source: str,
    draft: TranslationAttempt | None,
    target_language: str,
    attempts: list[TranslationAttempt],
    errors: list[str],
) -> TranslationAttempt | None:
    """Use SSD Qwen2.5-3B as the last local medical refinement stage."""

    if draft is None or not str(draft.text or "").strip():
        return None
    if not _model3_available(engine):
        return None

    backend = _model3(engine)
    try:
        text = backend.refine(
            source,
            draft.text,
            target_language,
        )
        attempt = cascade._quality_attempt(
            engine,
            backend.name,
            source,
            text,
            target_language,
        )
        attempts.append(attempt)
        return attempt
    except Exception as exc:
        errors.append(f"Qwen-model3: {type(exc).__name__}: {exc}")
        return None


def _run_local_cascade(
    engine: MultiModelTranslationEngine,
    source: str,
    target_language: str,
    attempts: list[TranslationAttempt],
    errors: list[str],
) -> tuple[TranslationAttempt | None, str]:
    """Run model1 -> conditional HY-MT -> conditional local Qwen model3.

    The normal fast path stops escalating as soon as a local stage passes its
    medical-quality gate. Qwen model3 is reserved for weak/problematic units,
    which avoids paying a 3B-generation latency on every paragraph. Set
    ``PHOENIX_MODEL3_ALWAYS=1`` only when an all-segment model3 audit is desired.
    """

    always_model3 = _model3_always()
    if not always_model3:
        _report_model3_fast_mode(engine)

    model1, model1_passed = _run_model1(
        engine,
        source,
        target_language,
        attempts,
        errors,
    )

    if model1_passed and not always_model3:
        return model1, "model1"

    base: TranslationAttempt | None = model1
    base_stage = "model1" if model1_passed else "model1_failed"

    if not model1_passed:
        model2 = cascade._run_model2(
            engine,
            source,
            model1,
            target_language,
            attempts,
            errors,
        )
        if (
            model2 is not None
            and model2.quality.ok
            and float(model2.quality.score) >= cascade.MODEL2_ACCEPT_SCORE
        ):
            if not always_model3:
                return model2, "model2"
            base = model2
            base_stage = "model2"
        else:
            base = _best_attempt(
                item for item in (model1, model2) if item is not None
            )
            base_stage = "model2_failed" if model2 is not None else "model1_failed"

    if base is None:
        return None, base_stage

    model3 = _run_model3(
        engine,
        source,
        base,
        target_language,
        attempts,
        errors,
    )
    if (
        model3 is not None
        and model3.quality.ok
        and float(model3.quality.score) >= MODEL3_ACCEPT_SCORE
    ):
        return model3, f"{base_stage}_model3"

    # A failed model3 refinement must never replace a safer prior draft. The
    # validator decides whether its output is eligible; otherwise Smart2 receives
    # the best pre-model3 local draft.
    return base, base_stage


def _api_polish_local_draft(
    engine: MultiModelTranslationEngine,
    source: str,
    local_draft: TranslationAttempt,
    local_stage: str,
    target_language: str,
    attempts: list[TranslationAttempt],
    errors: list[str],
) -> TranslationDecision | None:
    """Use Smart2 only as a final editor of the already translated local draft."""

    if not hybrid._smart_available(engine):
        return None

    reasons = tuple(
        (*local_draft.quality.reasons, _API_FINAL_POLISH_REASON)
    )
    try:
        polished = engine.qwen.retry_translation(
            source,
            local_draft.text,
            reasons,
            target_language,
        )
        first = cascade._quality_attempt(
            engine,
            f"{engine.qwen.name}_{local_stage}_final_polish_1",
            source,
            polished,
            target_language,
        )
        attempts.append(first)
        if first.quality.ok:
            return TranslationDecision(
                text=first.text,
                backend=first.backend,
                quality=first.quality,
                needs_review=False,
                attempts=tuple(attempts),
            )

        # One bounded correction pass is enough. The API is an editor here,
        # not another full translation tier, which keeps paid work predictable.
        corrected = engine.qwen.retry_translation(
            source,
            first.text or local_draft.text,
            tuple((*first.quality.reasons, _API_FINAL_POLISH_REASON)),
            target_language,
        )
        second = cascade._quality_attempt(
            engine,
            f"{engine.qwen.name}_{local_stage}_final_polish_2",
            source,
            corrected,
            target_language,
        )
        attempts.append(second)
        if second.quality.ok:
            return TranslationDecision(
                text=second.text,
                backend=second.backend,
                quality=second.quality,
                needs_review=False,
                attempts=tuple(attempts),
            )
    except Exception as exc:
        errors.append(f"Smart2-final-polish: {type(exc).__name__}: {exc}")
    return None


def _local_draft_accepted(local_draft: TranslationAttempt, local_stage: str) -> bool:
    if not local_draft.quality.ok:
        return False
    score = float(local_draft.quality.score)
    if local_stage.endswith("_model3"):
        return score >= MODEL3_ACCEPT_SCORE
    if local_stage == "model1":
        return score >= cascade.MODEL1_ACCEPT_SCORE
    if local_stage == "model2":
        return score >= cascade.MODEL2_ACCEPT_SCORE
    return False


def _translate(
    self: MultiModelTranslationEngine,
    source: str,
    target_language: str = "中文",
    *,
    smart_level: str = "smart1",
) -> TranslationDecision:
    level = _normalize_smart_level(smart_level)
    if level != "smart2":
        previous = getattr(self, "_phoenix_cascade_v2_previous_translate")
        return previous(source, target_language, smart_level=level)

    source = str(source or "").strip()
    attempts: list[TranslationAttempt] = []
    errors: list[str] = []

    local_draft, local_stage = _run_local_cascade(
        self,
        source,
        target_language,
        attempts,
        errors,
    )

    if local_draft is not None:
        # Smart2 remains the final editor when available, but the expensive local
        # model3 stage now runs only for translation units that actually need it.
        polished = _api_polish_local_draft(
            self,
            source,
            local_draft,
            local_stage,
            target_language,
            attempts,
            errors,
        )
        if polished is not None:
            return polished

        if _local_draft_accepted(local_draft, local_stage):
            return TranslationDecision(
                text=local_draft.text,
                backend=local_draft.backend,
                quality=local_draft.quality,
                needs_review=False,
                attempts=tuple(attempts),
            )
        return hybrid._reviewable_local_fallback(source, local_draft, attempts)

    # No local translator is usable. Only then is a source-only API translation
    # allowed, preserving functionality without changing the normal paid path.
    if hybrid._smart_available(self):
        try:
            translated = self.qwen.translate(
                source,
                target_language,
                smart_level="smart2",
            )
            first = cascade._quality_attempt(
                self,
                f"{self.qwen.name}_source_only_emergency",
                source,
                translated,
                target_language,
            )
            attempts.append(first)
            if first.quality.ok:
                return TranslationDecision(
                    text=first.text,
                    backend=first.backend,
                    quality=first.quality,
                    needs_review=False,
                    attempts=tuple(attempts),
                )
            return TranslationDecision(
                text=first.text,
                backend=first.backend,
                quality=first.quality,
                needs_review=True,
                attempts=tuple(attempts),
            )
        except Exception as exc:
            errors.append(f"Smart2-source-only: {type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("四级翻译链全部失败: " + " | ".join(errors))
    raise RuntimeError("没有可用的一级本地模型、HY-MT二级模型、Qwen三级模型或Smart2 API。")


def _translate_segments(
    self: MultiModelTranslationEngine,
    sources: list[str],
    target_language: str = "中文",
    *,
    smart_level: str = "smart2",
) -> tuple[TranslationDecision, ...]:
    values = [str(value or "").strip() for value in sources]
    if not values:
        return ()
    level = _normalize_smart_level(smart_level)
    if level != "smart2":
        previous = getattr(self, "_phoenix_cascade_v2_previous_translate_segments")
        return previous(values, target_language, smart_level=level)
    return tuple(
        _translate(self, value, target_language, smart_level="smart2")
        for value in values
    )


def _local_only_backend(name: str) -> bool:
    value = str(name or "").strip()
    return (
        value in LEGACY_PREVIEW_BACKEND_NAMES
        or value.startswith("hymt15_1p8b")
        or value.startswith("qwen_local_medical_model3")
        or value.startswith("local_guarded_review:")
        or value.startswith("local_source_preserved_review:")
        or value == "failed_preserve_source"
    )


def _office_unit_needs_refinement(payload: dict) -> bool:
    rows = [
        row for row in (payload.get("translations") or ())
        if isinstance(row, dict)
    ]
    if not rows:
        return False
    backends = [str(row.get("backend", "") or "") for row in rows]
    # Once an API-final-polish backend is present, the local draft has already
    # been upgraded and should remain resumable.
    if any(
        name.startswith("qwen35_medical_translation")
        and "final_polish" in name
        for name in backends
    ):
        return False
    return any(_local_only_backend(name) for name in backends)


def _patch_resume_and_cache_policy() -> None:
    # If a document was completed locally while offline, reconnecting the API
    # must invalidate those local-only checkpoints so they receive final polish.
    hybrid._is_local_only_backend = _local_only_backend
    hybrid._office_unit_needs_refinement = _office_unit_needs_refinement

    try:
        from . import translation_runtime_adapter as runtime_adapter

        def cacheable(runtime, decision) -> bool:
            if bool(getattr(decision, "needs_review", False)):
                return False
            if runtime.smart_level != "smart2":
                return True
            backend = str(getattr(decision, "backend", "") or "")
            # Smart2 tasks cache only API-polished results. Local-only results
            # must be recomputed when the API becomes available later.
            return (
                backend.startswith("qwen35_medical_translation")
                and ("final_polish" in backend or "source_only_emergency" in backend)
            )

        runtime_adapter._cacheable_decision = cacheable
    except Exception:
        pass


def _unload(self: MultiModelTranslationEngine) -> None:
    previous = getattr(self, "_phoenix_cascade_v2_previous_unload")
    try:
        previous()
    finally:
        backend = getattr(self, "_phoenix_qwen_model3", None)
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                pass


def install() -> None:
    cls = MultiModelTranslationEngine
    if bool(getattr(cls, "_phoenix_translation_cascade_v2_installed", False)):
        return

    cls._phoenix_cascade_v2_previous_translate = cls.translate
    cls._phoenix_cascade_v2_previous_translate_segments = cls.translate_segments
    cls._phoenix_cascade_v2_previous_unload = cls.unload
    cls.translate = _translate
    cls.translate_segments = _translate_segments
    cls.unload = _unload
    cls._phoenix_translation_cascade_v2_installed = True

    _patch_resume_and_cache_policy()
