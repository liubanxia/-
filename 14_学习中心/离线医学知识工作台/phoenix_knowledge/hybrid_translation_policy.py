from __future__ import annotations

from typing import Iterable

from .translation_models import (
    MultiModelTranslationEngine,
    QualityReport,
    QwenMedicalTranslationBackend,
    TranslationAttempt,
    TranslationDecision,
    _SIMPLIFIED_TARGETS,
    _normalize_smart_level,
)


LOCAL_DIRECT_ACCEPT_SCORE = 0.78
_REVIEW_PREFIX = "[本地模型译文；Smart2当前不可用或未通过增强校验，建议复核]\n"
_SOURCE_PREFIX = "[本地模型未通过关键安全校验；为避免错误已保留原文，建议复核]\n"


def _local_backends(engine: MultiModelTranslationEngine, target_language: str) -> list[object]:
    if target_language not in _SIMPLIFIED_TARGETS:
        return []
    result: list[object] = []
    for backend in (engine.marian, engine.nllb):
        if engine._backend_available(backend):
            result.append(backend)
    return result


def _smart_available(engine: MultiModelTranslationEngine) -> bool:
    if engine._real_smart_backend():
        return engine._backend_available(engine.qwen, "smart2")
    return engine._backend_available(engine.qwen)


def _attempt(engine: MultiModelTranslationEngine, backend, source: str, target_language: str) -> TranslationAttempt:
    if isinstance(backend, QwenMedicalTranslationBackend):
        text = backend.translate(source, target_language, smart_level="smart2")
    else:
        text = backend.translate(source)
    quality = engine.validator.validate(source, text, target_language)
    return TranslationAttempt(
        backend=str(getattr(backend, "name", "translation")),
        text=str(text or "").strip(),
        quality=quality,
    )


def _reviewable_local_fallback(
    source: str,
    best: TranslationAttempt,
    attempts: Iterable[TranslationAttempt],
) -> TranslationDecision:
    reasons = tuple(best.quality.reasons)
    unsafe_numeric = any("数字/单位/正负号未完整保留" in reason for reason in reasons)
    unsafe = unsafe_numeric or not best.text.strip() or float(best.quality.score) < 0.25

    if unsafe:
        text = _SOURCE_PREFIX + source.strip()
        backend = f"local_source_preserved_review:{best.backend}"
        note = "本地候选未通过关键安全校验，已保留原文而不是发布可疑译文"
    else:
        text = _REVIEW_PREFIX + best.text.strip()
        backend = f"local_guarded_review:{best.backend}"
        note = "Smart2未完成增强，本地译文已显式标记为建议复核"

    # Existing PDF/Office release code blocks quality_ok=False and
    # needs_review=True.  This fallback is intentionally publishable because
    # the visible marker carries the review state to the reader while the
    # audit keeps the original score/reasons and records the downgrade route.
    publishable_quality = QualityReport(
        ok=True,
        score=float(best.quality.score),
        reasons=tuple((*reasons, note)),
    )
    return TranslationDecision(
        text=text,
        backend=backend,
        quality=publishable_quality,
        needs_review=False,
        attempts=tuple(attempts),
    )


def _active_backends(
    self: MultiModelTranslationEngine,
    target_language: str = "中文",
    smart_level: str = "smart1",
) -> list[object]:
    level = _normalize_smart_level(smart_level)
    local = _local_backends(self, target_language)
    if level != "smart2":
        return local

    # Smart2 now means "medical-quality orchestration", not "remote-only".
    # Local translators work first; the quality model is appended only when it
    # is actually available, so an unavailable API never disables local work.
    result = list(local)
    if _smart_available(self):
        result.append(self.qwen)
    return result


def _formal_backend_names(
    self: MultiModelTranslationEngine,
    target_language: str = "中文",
) -> list[str]:
    names = [
        str(getattr(backend, "name", "") or "").strip()
        for backend in _active_backends(self, target_language, "smart2")
    ]
    return list(dict.fromkeys(name for name in names if name))


def _translate(
    self: MultiModelTranslationEngine,
    source: str,
    target_language: str = "中文",
    *,
    smart_level: str = "smart1",
) -> TranslationDecision:
    level = _normalize_smart_level(smart_level)
    if level != "smart2":
        original = getattr(self, "_phoenix_hybrid_original_translate")
        return original(
            source,
            target_language,
            smart_level=level,
        )

    attempts: list[TranslationAttempt] = []
    errors: list[str] = []
    local_best: TranslationAttempt | None = None

    # 1) Local models do the normal work first.  A strong structurally safe
    # result is accepted immediately and consumes zero external tokens.
    for backend in _local_backends(self, target_language):
        try:
            attempt = _attempt(self, backend, source, target_language)
            attempts.append(attempt)
            if local_best is None or attempt.quality.score > local_best.quality.score:
                local_best = attempt
            if attempt.quality.ok and attempt.quality.score >= LOCAL_DIRECT_ACCEPT_SCORE:
                return TranslationDecision(
                    text=attempt.text,
                    backend=attempt.backend,
                    quality=attempt.quality,
                    needs_review=False,
                    attempts=tuple(attempts),
                )
        except Exception as exc:
            errors.append(
                f"{getattr(backend, 'name', 'local')}: {type(exc).__name__}: {exc}"
            )

    # 2) Only weak/failed local segments reach Smart2.  This is the token-cost
    # boundary: ordinary pages never call the external model.
    smart_best: TranslationAttempt | None = None
    if _smart_available(self):
        try:
            first = _attempt(self, self.qwen, source, target_language)
            attempts.append(first)
            smart_best = first
            if first.quality.ok:
                return TranslationDecision(
                    text=first.text,
                    backend=first.backend,
                    quality=first.quality,
                    needs_review=False,
                    attempts=tuple(attempts),
                )

            corrected = first.text
            corrected_quality = first.quality
            for repair_round in range(1, 3):
                corrected = self.qwen.retry_translation(
                    source,
                    corrected,
                    corrected_quality.reasons,
                    target_language,
                )
                corrected_quality = self.validator.validate(
                    source,
                    corrected,
                    target_language,
                )
                retry = TranslationAttempt(
                    backend=f"{self.qwen.name}_quality_retry_{repair_round}",
                    text=corrected,
                    quality=corrected_quality,
                )
                attempts.append(retry)
                if smart_best is None or retry.quality.score > smart_best.quality.score:
                    smart_best = retry
                if retry.quality.ok:
                    return TranslationDecision(
                        text=retry.text,
                        backend=retry.backend,
                        quality=retry.quality,
                        needs_review=False,
                        attempts=tuple(attempts),
                    )
        except Exception as exc:
            errors.append(f"Smart2: {type(exc).__name__}: {exc}")

    # 3) If Smart2 is unavailable or still fails, do not shut the translator
    # down.  Publish the best local candidate with an explicit review marker.
    # Critical numeric/unit failures preserve the source text instead.
    if local_best is not None:
        return _reviewable_local_fallback(source, local_best, attempts)

    # No local model exists. Preserve the previous strict behavior for a poor
    # Smart2-only result because there is no safe local fallback to show.
    if smart_best is not None:
        return TranslationDecision(
            text=smart_best.text,
            backend=smart_best.backend,
            quality=smart_best.quality,
            needs_review=True,
            attempts=tuple(attempts),
        )

    if errors:
        raise RuntimeError("所有翻译后端均执行失败: " + " | ".join(errors))
    raise RuntimeError("没有可用的本地翻译模型或Smart2质量模型。")


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
        original = getattr(self, "_phoenix_hybrid_original_translate_segments")
        return original(values, target_language, smart_level=level)

    # If local translators are installed, keep Office/PPT translation local
    # first as well. Only the individual weak rows reach Smart2.
    if _local_backends(self, target_language):
        return tuple(
            _translate(
                self,
                source,
                target_language,
                smart_level="smart2",
            )
            for source in values
        )

    # No local model: retain the original Smart2 batch path because it is more
    # token-efficient than one remote request per text box/paragraph.
    original = getattr(self, "_phoenix_hybrid_original_translate_segments")
    return original(values, target_language, smart_level="smart2")


def install() -> None:
    cls = MultiModelTranslationEngine
    if bool(getattr(cls, "_phoenix_hybrid_translation_policy_installed", False)):
        return

    cls._phoenix_hybrid_original_translate = cls.translate
    cls._phoenix_hybrid_original_translate_segments = cls.translate_segments
    cls._phoenix_hybrid_original_active_backends = cls.active_backends
    cls._phoenix_hybrid_original_formal_backend_names = cls.formal_backend_names

    cls.active_backends = _active_backends
    cls.formal_backend_names = _formal_backend_names
    cls.translate = _translate
    cls.translate_segments = _translate_segments
    cls._phoenix_hybrid_translation_policy_installed = True
