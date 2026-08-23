from __future__ import annotations

from .translation_models import TranslationAttempt, TranslationDecision
from .translation_review_pipeline import MedicalTranslationReviewPipeline


_REVIEW_SEPARATOR = "\n<<<PHOENIX_SEGMENT_BOUNDARY>>>\n"
_INSTALLED = False


def _stage_payload(item) -> dict:
    return {
        "stage": item.stage,
        "backend": item.backend,
        "changed": bool(item.changed),
        "passed": bool(item.passed),
        "accepted": bool(item.accepted),
        "quality_score": round(float(item.quality_score), 4),
        "reasons": list(item.reasons),
        "error": item.error,
    }


def _review_pdf_page(
    self,
    source_text: str,
    page_number: int,
    target_language: str,
    *,
    smart_level: str = "smart1",
    status=None,
):
    previous = getattr(type(self), "_phoenix_review_previous_translate_page")
    translated, audit = previous(
        self,
        source_text,
        page_number,
        target_language,
        smart_level=smart_level,
        status=status,
    )
    source = str(source_text or "").strip()
    draft = str(translated or "").strip()
    if not source or not draft:
        return translated, audit

    if status:
        status(f"第 {page_number} 页：翻译完成，开始四级整页医学复核……")
    try:
        reviewed, stages = MedicalTranslationReviewPipeline(self.engine).run(
            source,
            draft,
            target_language,
            label=f"PDF第{page_number}页",
        )
    except Exception as exc:
        payload = dict(audit or {})
        payload["page_review_error"] = f"{type(exc).__name__}: {exc}"
        return translated, payload

    final_quality = self.engine.validator.validate(source, reviewed, target_language)
    payload = dict(audit or {})
    payload["pre_review_warning_count"] = int(payload.get("warning_count", 0) or 0)
    payload["review_stages"] = [_stage_payload(item) for item in stages]
    payload["final_review_quality"] = {
        "ok": bool(final_quality.ok),
        "score": round(float(final_quality.score), 4),
        "reasons": list(final_quality.reasons),
    }
    # A successful whole-page review repairs earlier per-chunk warnings. If the
    # final scan still fails, formal publication remains blocked as before.
    payload["warning_count"] = (
        0
        if final_quality.ok
        else max(1, int(payload.get("warning_count", 0) or 0))
    )
    if status:
        status(
            f"第 {page_number} 页：四级整页复核完成 | "
            f"最终校验={'PASS' if final_quality.ok else 'REVIEW'}"
        )
    return reviewed, payload


def _review_office_sources(self, sources: list[str], target_language: str):
    previous = getattr(type(self), "_phoenix_review_previous_translate_sources")
    decisions = list(previous(self, sources, target_language))
    if not sources or len(decisions) != len(sources):
        return decisions

    source_bundle = _REVIEW_SEPARATOR.join(str(value or "").strip() for value in sources)
    draft_bundle = _REVIEW_SEPARATOR.join(
        str(getattr(decision, "text", "") or "").strip()
        for decision in decisions
    )
    if not source_bundle.strip() or not draft_bundle.strip():
        return decisions

    try:
        reviewed_bundle, stages = MedicalTranslationReviewPipeline(self.engine).run(
            source_bundle,
            draft_bundle,
            target_language,
            separator=_REVIEW_SEPARATOR,
            expected_segments=len(sources),
            label=f"Office整单元批次({len(sources)}段)",
        )
    except Exception as exc:
        print(
            f"[Phoenix][复核] Office整单元复核失败，保留翻译阶段结果: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return decisions

    reviewed_parts = reviewed_bundle.split(_REVIEW_SEPARATOR)
    if len(reviewed_parts) != len(decisions):
        print("[Phoenix][复核] Office段落边界变化，已丢弃本次整单元改写。", flush=True)
        return decisions

    stage_names = ",".join(item.stage for item in stages)
    result: list[TranslationDecision] = []
    for source, decision, reviewed in zip(sources, decisions, reviewed_parts):
        reviewed = str(reviewed or "").strip()
        old_text = str(getattr(decision, "text", "") or "").strip()
        old_quality = getattr(decision, "quality", None)
        new_quality = self.engine.validator.validate(source, reviewed, target_language)

        # Never let page-level editing make a previously safe segment invalid.
        if (
            old_quality is not None
            and bool(getattr(old_quality, "ok", False))
            and (
                not new_quality.ok
                or float(new_quality.score) < float(getattr(old_quality, "score", 0.0))
            )
        ):
            final_text = old_text
            final_quality = old_quality
        else:
            final_text = reviewed or old_text
            final_quality = new_quality

        review_attempt = TranslationAttempt(
            backend=f"page_review:{stage_names}",
            text=final_text,
            quality=final_quality,
        )
        attempts = tuple(getattr(decision, "attempts", ()) or ()) + (review_attempt,)
        result.append(
            TranslationDecision(
                text=final_text,
                backend=f"{getattr(decision, 'backend', 'translation')}|reviewed",
                quality=final_quality,
                needs_review=not bool(final_quality.ok),
                attempts=attempts,
            )
        )

    print(
        f"[Phoenix][复核] Office整单元批次完成 | 段数={len(result)} | "
        f"未通过={sum(1 for item in result if item.needs_review)}",
        flush=True,
    )
    return result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .translator import PDFTranslator
    from .office_translation import OfficeDocumentTranslator

    if not hasattr(PDFTranslator, "_phoenix_review_previous_translate_page"):
        PDFTranslator._phoenix_review_previous_translate_page = PDFTranslator._translate_page
        PDFTranslator._translate_page = _review_pdf_page

    if not hasattr(OfficeDocumentTranslator, "_phoenix_review_previous_translate_sources"):
        OfficeDocumentTranslator._phoenix_review_previous_translate_sources = (
            OfficeDocumentTranslator._translate_sources
        )
        OfficeDocumentTranslator._translate_sources = _review_office_sources
