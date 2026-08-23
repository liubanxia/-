from __future__ import annotations

from typing import Iterable

from . import hybrid_translation_policy as hybrid
from .hymt_translation_backend import HYMTMedicalTranslationBackend
from .translation_models import (
    MultiModelTranslationEngine,
    QwenMedicalTranslationBackend,
    TranslationAttempt,
    TranslationDecision,
    _normalize_smart_level,
)


MODEL1_ACCEPT_SCORE = 0.78
MODEL2_ACCEPT_SCORE = 0.62
_API_REFINEMENT_REASON = (
    "前两级本地翻译仍未达到质量门槛。请严格对照英文原文，在保留数字、单位、正负号、"
    "侧别、否定关系、分级、医学缩写和图表编号的前提下，对现有中文译文做最终医学精修；"
    "不得总结、删减、扩写或只润色而不核对原文。"
)


def _model2(engine: MultiModelTranslationEngine) -> HYMTMedicalTranslationBackend:
    backend = getattr(engine, "_phoenix_hymt_model2", None)
    if backend is None:
        backend = HYMTMedicalTranslationBackend(engine.paths)
        engine._phoenix_hymt_model2 = backend
    return backend


def _model2_available(engine: MultiModelTranslationEngine) -> bool:
    try:
        return _model2(engine).available()
    except Exception:
        return False


def _quality_attempt(
    engine: MultiModelTranslationEngine,
    backend_name: str,
    source: str,
    translated: str,
    target_language: str,
) -> TranslationAttempt:
    text = str(translated or "").strip()
    quality = engine.validator.validate(source, text, target_language)
    return TranslationAttempt(
        backend=backend_name,
        text=text,
        quality=quality,
    )


def _best_attempt(attempts: Iterable[TranslationAttempt]) -> TranslationAttempt | None:
    values = [attempt for attempt in attempts if str(attempt.text or "").strip()]
    if not values:
        return None
    return max(values, key=lambda item: float(item.quality.score))


def _run_model2(
    engine: MultiModelTranslationEngine,
    source: str,
    draft: TranslationAttempt | None,
    target_language: str,
    attempts: list[TranslationAttempt],
    errors: list[str],
) -> TranslationAttempt | None:
    if not _model2_available(engine):
        return None
    backend = _model2(engine)
    try:
        if draft is not None and draft.text.strip():
            text = backend.refine(source, draft.text, target_language)
            name = backend.name
        else:
            text = backend.translate(source, target_language)
            name = f"{backend.name}:source"
        attempt = _quality_attempt(
            engine,
            name,
            source,
            text,
            target_language,
        )
        attempts.append(attempt)
        return attempt
    except Exception as exc:
        errors.append(f"HY-MT-model2: {type(exc).__name__}: {exc}")
        return None


def _run_api_final_refine(
    engine: MultiModelTranslationEngine,
    source: str,
    draft: TranslationAttempt | None,
    target_language: str,
    attempts: list[TranslationAttempt],
    errors: list[str],
) -> TranslationDecision | None:
    if not hybrid._smart_available(engine):
        return None

    try:
        if draft is None or not draft.text.strip():
            translated = engine.qwen.translate(
                source,
                target_language,
                smart_level="smart2",
            )
            first = _quality_attempt(
                engine,
                engine.qwen.name,
                source,
                translated,
                target_language,
            )
        else:
            reasons = tuple((*draft.quality.reasons, _API_REFINEMENT_REASON))
            translated = engine.qwen.retry_translation(
                source,
                draft.text,
                reasons,
                target_language,
            )
            first = _quality_attempt(
                engine,
                f"{engine.qwen.name}_model2_draft_refine_1",
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

        corrected = engine.qwen.retry_translation(
            source,
            first.text,
            tuple((*first.quality.reasons, _API_REFINEMENT_REASON)),
            target_language,
        )
        second = _quality_attempt(
            engine,
            f"{engine.qwen.name}_model2_draft_refine_2",
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
        errors.append(f"Smart2-final-refine: {type(exc).__name__}: {exc}")
    return None


def _active_backends(
    self: MultiModelTranslationEngine,
    target_language: str = "中文",
    smart_level: str = "smart1",
) -> list[object]:
    level = _normalize_smart_level(smart_level)
    if level != "smart2":
        return list(hybrid._local_backends(self, target_language))

    result = list(hybrid._local_backends(self, target_language))
    if _model2_available(self):
        result.append(_model2(self))
    if hybrid._smart_available(self):
        result.append(self.qwen)
    return result


def _formal_backend_names(
    self: MultiModelTranslationEngine,
    target_language: str = "中文",
) -> list[str]:
    names = [
        str(getattr(backend, "name", "") or "").strip()
        for backend in self.active_backends(target_language, "smart2")
    ]
    return list(dict.fromkeys(name for name in names if name))


def _available_backends(self: MultiModelTranslationEngine) -> list[str]:
    original = getattr(self, "_phoenix_hymt_previous_available_backends")
    names = list(original())
    if _model2_available(self):
        names.append(_model2(self).name)
    return list(dict.fromkeys(str(name) for name in names if str(name)))


def _translate(
    self: MultiModelTranslationEngine,
    source: str,
    target_language: str = "中文",
    *,
    smart_level: str = "smart1",
) -> TranslationDecision:
    level = _normalize_smart_level(smart_level)
    if level != "smart2":
        previous = getattr(self, "_phoenix_hymt_previous_translate")
        return previous(source, target_language, smart_level=level)

    source = str(source or "").strip()
    attempts: list[TranslationAttempt] = []
    errors: list[str] = []
    model1_best: TranslationAttempt | None = None

    # Stage 1: existing lightweight local translators.
    for backend in hybrid._local_backends(self, target_language):
        try:
            attempt = hybrid._attempt(self, backend, source, target_language)
            attempts.append(attempt)
            if model1_best is None or attempt.quality.score > model1_best.quality.score:
                model1_best = attempt
            if attempt.quality.ok and attempt.quality.score >= MODEL1_ACCEPT_SCORE:
                return TranslationDecision(
                    text=attempt.text,
                    backend=attempt.backend,
                    quality=attempt.quality,
                    needs_review=False,
                    attempts=tuple(attempts),
                )
        except Exception as exc:
            errors.append(
                f"model1-{getattr(backend, 'name', 'local')}: {type(exc).__name__}: {exc}"
            )

    # Stage 2: HY-MT receives the English source plus the best model-1 draft.
    model2_attempt = _run_model2(
        self,
        source,
        model1_best,
        target_language,
        attempts,
        errors,
    )
    if (
        model2_attempt is not None
        and model2_attempt.quality.ok
        and model2_attempt.quality.score >= MODEL2_ACCEPT_SCORE
    ):
        return TranslationDecision(
            text=model2_attempt.text,
            backend=model2_attempt.backend,
            quality=model2_attempt.quality,
            needs_review=False,
            attempts=tuple(attempts),
        )

    # Stage 3: API is the final fallback only. It refines the best local draft
    # rather than discarding previous work and translating blindly from scratch.
    local_best = _best_attempt(
        attempt
        for attempt in (model1_best, model2_attempt)
        if attempt is not None
    )
    api_result = _run_api_final_refine(
        self,
        source,
        local_best,
        target_language,
        attempts,
        errors,
    )
    if api_result is not None:
        return api_result

    if local_best is not None:
        return hybrid._reviewable_local_fallback(source, local_best, attempts)

    if errors:
        raise RuntimeError("三级翻译链全部失败: " + " | ".join(errors))
    raise RuntimeError("没有可用的一级本地模型、HY-MT二级模型或Smart2 API。")


def _translate_segments(
    self: MultiModelTranslationEngine,
    sources: list[str],
    target_language: str = "中文",
    *,
    smart_level: str = "smart2",
) -> tuple[TranslationDecision, ...]:
    values = [str(source or "").strip() for source in sources]
    if not values:
        return ()
    level = _normalize_smart_level(smart_level)
    if level != "smart2":
        previous = getattr(self, "_phoenix_hymt_previous_translate_segments")
        return previous(values, target_language, smart_level=level)

    # If no local stage exists, preserve the previous Smart2 batch route.
    if not hybrid._local_backends(self, target_language) and not _model2_available(self):
        previous = getattr(self, "_phoenix_hymt_previous_translate_segments")
        return previous(values, target_language, smart_level="smart2")

    return tuple(
        _translate(self, source, target_language, smart_level="smart2")
        for source in values
    )


def _unload(self: MultiModelTranslationEngine) -> None:
    previous = getattr(self, "_phoenix_hymt_previous_unload")
    try:
        previous()
    finally:
        backend = getattr(self, "_phoenix_hymt_model2", None)
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                pass


def _patch_resume_policy() -> None:
    # With the three-stage cascade, a strong model-1 or HY-MT result is final.
    # Only explicitly guarded/review fallbacks should be invalidated when an API
    # later becomes available.
    def local_only(name: str) -> bool:
        value = str(name or "").strip()
        return (
            value.startswith("local_guarded_review:")
            or value.startswith("local_source_preserved_review:")
            or value == "failed_preserve_source"
        )

    def office_needs_refinement(payload: dict) -> bool:
        rows = [
            row for row in (payload.get("translations") or ())
            if isinstance(row, dict)
        ]
        if not rows:
            return False
        backends = [str(row.get("backend", "") or "") for row in rows]
        return any(local_only(name) for name in backends)

    hybrid._is_local_only_backend = local_only
    hybrid._office_unit_needs_refinement = office_needs_refinement


def _patch_runtime_cache() -> None:
    try:
        from . import translation_runtime_adapter as runtime_adapter

        def cacheable(runtime, decision) -> bool:
            if bool(getattr(decision, "needs_review", False)):
                return False
            backend = str(getattr(decision, "backend", "") or "")
            if backend.startswith(("local_guarded_review:", "local_source_preserved_review:")):
                return False
            if backend == "failed_preserve_source":
                return False
            if runtime.smart_level != "smart2":
                return True
            return (
                backend in {"marian_en_zh", "nllb_600m_en_zh"}
                or backend.startswith("hymt15_1p8b_refine")
                or backend.startswith("qwen35_medical_translation")
            )

        runtime_adapter._cacheable_decision = cacheable
    except Exception:
        pass


def install() -> None:
    cls = MultiModelTranslationEngine
    if bool(getattr(cls, "_phoenix_hymt_cascade_installed", False)):
        return

    cls._phoenix_hymt_previous_translate = cls.translate
    cls._phoenix_hymt_previous_translate_segments = cls.translate_segments
    cls._phoenix_hymt_previous_active_backends = cls.active_backends
    cls._phoenix_hymt_previous_formal_backend_names = cls.formal_backend_names
    cls._phoenix_hymt_previous_available_backends = cls.available_backends
    cls._phoenix_hymt_previous_unload = cls.unload

    cls.active_backends = _active_backends
    cls.formal_backend_names = _formal_backend_names
    cls.available_backends = _available_backends
    cls.translate = _translate
    cls.translate_segments = _translate_segments
    cls.unload = _unload
    cls._phoenix_hymt_cascade_installed = True

    _patch_resume_policy()
    _patch_runtime_cache()
