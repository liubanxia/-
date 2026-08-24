from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .translation_models import (
    LEGACY_PREVIEW_BACKEND_NAMES,
    MultiModelTranslationEngine,
    QualityReport,
    QwenMedicalTranslationBackend,
    TranslationAttempt,
    TranslationDecision,
    _normalize_smart_level,
)


SMART2_REFINEMENT_EPOCH = 1
_REFINEMENT_REASON = (
    "这是本地模型已经完成的医学初译。请不要重新摘要；请在逐句保留原意、数字、单位、"
    "正负号、侧别、否定关系、分级和医学缩写的前提下，对现有中文译文进行二次医学精修，"
    "重点修正术语、语序、漏译、生硬直译和上下文表达。"
)


def _local_backends(engine: MultiModelTranslationEngine, target_language: str) -> list[object]:
    del engine, target_language
    # Smart1/legacy Marian and NLLB candidates are permanently excluded from
    # every formal or experimental medical route. HY-MT/model3 have their own
    # explicit stages; keeping this function empty also prevents an installed
    # legacy folder from silently re-entering Smart2 inventory.
    return []


def _smart_available(engine: MultiModelTranslationEngine) -> bool:
    backend = getattr(engine, "qwen", None)
    available = getattr(engine, "_backend_available", None)
    if backend is None or not callable(available):
        return False
    real = getattr(engine, "_real_smart_backend", None)
    try:
        if callable(real) and real():
            return bool(available(backend, "smart2"))
        return bool(available(backend))
    except Exception:
        return False


def _attempt(
    engine: MultiModelTranslationEngine,
    backend,
    source: str,
    target_language: str,
) -> TranslationAttempt:
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
    del source
    reasons = tuple(best.quality.reasons)
    note = "自动医学质量门未通过；禁止发布并等待可用模型自动重试"
    blocked_quality = QualityReport(
        ok=False,
        score=min(float(best.quality.score), 0.61),
        reasons=tuple((*reasons, note)),
    )
    return TranslationDecision(
        text=best.text.strip(),
        backend=f"blocked_local_candidate:{best.backend}",
        quality=blocked_quality,
        needs_review=True,
        attempts=tuple(attempts),
    )


def _refine_local_with_smart2(
    self: MultiModelTranslationEngine,
    source: str,
    local_best: TranslationAttempt,
    target_language: str,
    attempts: list[TranslationAttempt],
    errors: list[str],
) -> TranslationDecision | None:
    """Feed the local Chinese draft back into Smart2 for a real second pass."""

    if not _smart_available(self):
        return None

    draft = local_best.text
    reasons = tuple(local_best.quality.reasons) or (_REFINEMENT_REASON,)
    try:
        refined = self.qwen.retry_translation(
            source,
            draft,
            tuple((*reasons, _REFINEMENT_REASON)),
            target_language,
        )
        refined_quality = self.validator.validate(source, refined, target_language)
        refined_attempt = TranslationAttempt(
            backend=f"{self.qwen.name}_local_draft_refine_1",
            text=str(refined or "").strip(),
            quality=refined_quality,
        )
        attempts.append(refined_attempt)
        if refined_quality.ok:
            return TranslationDecision(
                text=refined_attempt.text,
                backend=refined_attempt.backend,
                quality=refined_quality,
                needs_review=False,
                attempts=tuple(attempts),
            )

        second = self.qwen.retry_translation(
            source,
            refined_attempt.text,
            tuple((*refined_quality.reasons, _REFINEMENT_REASON)),
            target_language,
        )
        second_quality = self.validator.validate(source, second, target_language)
        second_attempt = TranslationAttempt(
            backend=f"{self.qwen.name}_local_draft_refine_2",
            text=str(second or "").strip(),
            quality=second_quality,
        )
        attempts.append(second_attempt)
        if second_quality.ok:
            return TranslationDecision(
                text=second_attempt.text,
                backend=second_attempt.backend,
                quality=second_quality,
                needs_review=False,
                attempts=tuple(attempts),
            )
    except Exception as exc:
        errors.append(f"Smart2-local-refine: {type(exc).__name__}: {exc}")
    return None


def _active_backends(
    self: MultiModelTranslationEngine,
    target_language: str = "中文",
    smart_level: str = "smart1",
) -> list[object]:
    level = _normalize_smart_level(smart_level)
    local = _local_backends(self, target_language)
    if level != "smart2":
        return local

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
        return original(source, target_language, smart_level=level)

    attempts: list[TranslationAttempt] = []
    errors: list[str] = []
    local_best: TranslationAttempt | None = None
    smart_ready = _smart_available(self)

    # Local models always create the first-pass translation. When Smart2 is
    # connected we deliberately do NOT return even a strong local draft here:
    # the API must receive that translated draft and perform the requested
    # second-pass medical refinement.
    for backend in _local_backends(self, target_language):
        try:
            attempt = _attempt(self, backend, source, target_language)
            attempts.append(attempt)
            if local_best is None or attempt.quality.score > local_best.quality.score:
                local_best = attempt
            # Legacy local translators are excluded by _local_backends. Keep
            # this loop generic for adapter tests, but never publish its draft.
        except Exception as exc:
            errors.append(
                f"{getattr(backend, 'name', 'local')}: {type(exc).__name__}: {exc}"
            )

    # Smart2 connected + local draft available: refine the actual local Chinese
    # draft, rather than ignoring it and either reusing it unchanged or doing a
    # disconnected translation from source only.
    if local_best is not None and smart_ready:
        refined = _refine_local_with_smart2(
            self,
            source,
            local_best,
            target_language,
            attempts,
            errors,
        )
        if refined is not None:
            return refined

    # If there was no usable local draft, Smart2 still has a source-only path.
    smart_best: TranslationAttempt | None = None
    if smart_ready and local_best is None:
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

    if local_best is not None:
        return _reviewable_local_fallback(source, local_best, attempts)

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

    # With local models installed, each unique source is translated locally and
    # then refined by Smart2 when available. This guarantees that reconnecting
    # the API actually upgrades the existing local translation.
    if _local_backends(self, target_language):
        return tuple(
            _translate(self, source, target_language, smart_level="smart2")
            for source in values
        )

    original = getattr(self, "_phoenix_hybrid_original_translate_segments")
    return original(values, target_language, smart_level="smart2")


def _is_local_only_backend(name: str) -> bool:
    value = str(name or "").strip()
    return (
        value in LEGACY_PREVIEW_BACKEND_NAMES
        or value.startswith("blocked_local_candidate:")
        or value.startswith("local_guarded_review:")
        or value.startswith("local_source_preserved_review:")
        or value == "failed_preserve_source"
    )


def _pdf_audit_needs_refinement(payload: dict) -> bool:
    parts = payload.get("parts") or ()
    for part in parts:
        if not isinstance(part, dict):
            continue
        if _is_local_only_backend(str(part.get("backend", ""))):
            return True
    return False


def _invalidate_pdf_local_checkpoints(translator, pdf_path: Path, target_language: str) -> None:
    if not _smart_available(translator.engine):
        return
    try:
        from .pdf_parser import sha256_file

        source = Path(pdf_path).resolve()
        digest = sha256_file(source)
        _root, pages_root, audit_root, checkpoint, _final = translator._book_paths(
            source,
            digest,
            target_language,
        )
        changed = False
        for audit_file in audit_root.glob("*.json"):
            payload = translator._read_json(audit_file)
            if not _pdf_audit_needs_refinement(payload):
                continue
            (pages_root / f"{audit_file.stem}.txt").unlink(missing_ok=True)
            audit_file.unlink(missing_ok=True)
            changed = True
        if changed:
            state = translator._read_json(checkpoint)
            if state:
                state["status"] = "running"
                state["smart2_refinement_pending"] = True
                translator._write_json(checkpoint, state)
    except Exception:
        # Cache invalidation must never prevent a translation from starting.
        return


def _office_unit_needs_refinement(payload: dict) -> bool:
    if int(payload.get("smart2_refinement_epoch", 0) or 0) >= SMART2_REFINEMENT_EPOCH:
        return False
    rows = [row for row in (payload.get("translations") or ()) if isinstance(row, dict)]
    if not rows:
        return False
    backends = [str(row.get("backend", "") or "") for row in rows]
    if any(_is_local_only_backend(name) for name in backends):
        return True
    # Old document_cache rows do not record whether their source was a local or
    # Smart2 translation. Rebuild them once after Smart2 becomes available;
    # the epoch marker below prevents repeated rebuilds on later runs.
    if any(name == "document_cache" for name in backends):
        return True
    return not any(name.startswith("qwen35_medical_translation") for name in backends)


def _invalidate_office_local_checkpoints(translator, source_path: Path, target_language: str) -> tuple[Path, Path] | None:
    if not _smart_available(translator.engine):
        return None
    try:
        from .office_translation import _sha256_file

        source = Path(source_path).resolve()
        digest = _sha256_file(source)
        _root, units_root, previews_root, _checkpoint, _final = translator._task_paths(
            source,
            digest,
            target_language,
        )
        for unit_file in units_root.glob("*.json"):
            try:
                payload = json.loads(unit_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict) or not _office_unit_needs_refinement(payload):
                continue
            unit_file.unlink(missing_ok=True)
            (previews_root / f"{unit_file.stem}.txt").unlink(missing_ok=True)
        return units_root, previews_root
    except Exception:
        return None


def _mark_office_refinement_epoch(units_root: Path | None) -> None:
    if units_root is None or not units_root.is_dir():
        return
    for unit_file in units_root.glob("*.json"):
        try:
            payload = json.loads(unit_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        rows = [row for row in (payload.get("translations") or ()) if isinstance(row, dict)]
        if any(_is_local_only_backend(str(row.get("backend", ""))) for row in rows):
            continue
        payload["smart2_refinement_epoch"] = SMART2_REFINEMENT_EPOCH
        temp = unit_file.with_suffix(unit_file.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(unit_file)


def _install_resume_refinement_hooks() -> None:
    # When a document was translated locally while the API was unavailable,
    # reconnecting Smart2 must upgrade those saved checkpoints instead of
    # silently reusing them forever.
    try:
        from .translator import PDFTranslator

        if not bool(getattr(PDFTranslator, "_phoenix_smart2_refine_resume_installed", False)):
            original_pdf = PDFTranslator.translate_book

            def translate_book(self, pdf_path, *args, **kwargs):
                if not bool(kwargs.get("force_restart", False)):
                    _invalidate_pdf_local_checkpoints(
                        self,
                        Path(pdf_path),
                        str(kwargs.get("target_language", "中文")),
                    )
                return original_pdf(self, pdf_path, *args, **kwargs)

            PDFTranslator.translate_book = translate_book
            PDFTranslator._phoenix_smart2_refine_resume_installed = True
    except Exception:
        pass

    try:
        from .office_translation import OfficeDocumentTranslator

        if not bool(getattr(OfficeDocumentTranslator, "_phoenix_smart2_refine_resume_installed", False)):
            original_office = OfficeDocumentTranslator.translate_document

            def translate_document(self, source_path, *args, **kwargs):
                units_root = None
                if not bool(kwargs.get("force_restart", False)):
                    result = _invalidate_office_local_checkpoints(
                        self,
                        Path(source_path),
                        str(kwargs.get("target_language", "中文")),
                    )
                    units_root = result[0] if result else None
                output = original_office(self, source_path, *args, **kwargs)
                if _smart_available(self.engine):
                    if units_root is None:
                        try:
                            from .office_translation import _sha256_file

                            source = Path(source_path).resolve()
                            digest = _sha256_file(source)
                            _root, units_root, _previews, _checkpoint, _final = self._task_paths(
                                source,
                                digest,
                                str(kwargs.get("target_language", "中文")),
                            )
                        except Exception:
                            units_root = None
                    _mark_office_refinement_epoch(units_root)
                return output

            OfficeDocumentTranslator.translate_document = translate_document
            OfficeDocumentTranslator._phoenix_smart2_refine_resume_installed = True
    except Exception:
        pass


def install() -> None:
    cls = MultiModelTranslationEngine
    if not bool(getattr(cls, "_phoenix_hybrid_translation_policy_installed", False)):
        cls._phoenix_hybrid_original_translate = cls.translate
        cls._phoenix_hybrid_original_translate_segments = cls.translate_segments
        cls._phoenix_hybrid_original_active_backends = cls.active_backends
        cls._phoenix_hybrid_original_formal_backend_names = cls.formal_backend_names

        cls.active_backends = _active_backends
        cls.formal_backend_names = _formal_backend_names
        cls.translate = _translate
        cls.translate_segments = _translate_segments
        cls._phoenix_hybrid_translation_policy_installed = True

    _install_resume_refinement_hooks()
