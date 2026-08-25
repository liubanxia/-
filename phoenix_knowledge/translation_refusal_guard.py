from __future__ import annotations

from typing import Iterable


_INSTALLED = False

# Conservative refusal detection: ordinary medical prose may contain words such
# as “无法”, so generic negative wording alone never triggers the guard.
_STRONG_PHRASES = (
    "系统检测到您输入的内容可能涉及医疗或健康领域",
    "请提供具体的英文原文",
    "我无法直接处理",
    "无法直接处理",
    "无法协助处理",
    "无法帮助处理",
    "i can't help with medical or health",
    "i cannot help with medical or health",
    "i’m unable to help with medical or health",
    "i'm unable to help with medical or health",
    "please provide the specific english source",
)
_APOLOGY_PREFIXES = (
    "抱歉",
    "对不起",
    "很抱歉",
    "sorry",
    "i'm sorry",
    "i’m sorry",
)
_REFUSAL_TERMS = (
    "无法",
    "不能",
    "不可以",
    "can't",
    "cannot",
    "unable",
    "not able",
)
_CONTEXT_TERMS = (
    "医疗",
    "医学",
    "健康",
    "内容",
    "处理",
    "协助",
    "帮助",
    "medical",
    "health",
    "content",
    "assist",
    "help",
)


def looks_like_model_refusal(text: str) -> bool:
    value = " ".join(str(text or "").strip().split())
    if not value:
        return False
    lowered = value.lower()
    if any(phrase in lowered for phrase in _STRONG_PHRASES):
        return True

    starts_apology = any(lowered.startswith(prefix) for prefix in _APOLOGY_PREFIXES)
    if not starts_apology:
        return False
    has_refusal = any(term in lowered for term in _REFUSAL_TERMS)
    has_context = any(term in lowered for term in _CONTEXT_TERMS)
    return has_refusal and has_context


def _append_reason(existing: Iterable[str], reason: str) -> list[str]:
    result = [str(item) for item in existing if str(item).strip()]
    if reason not in result:
        result.append(reason)
    return result


def install() -> None:
    """Enforce a final fail-safe publication gate for medical translation.

    The shared validator rejects model refusal templates. Office delivery adds a
    stronger invariant at the last point before translated text is copied into
    PPTX/DOCX: only quality-approved, non-review output may replace the source.
    Every rejected candidate remains in the audit record but the deliverable
    preserves the original source text. This prevents refusal prose, malformed
    retries, and other unapproved model output from contaminating formal files.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .translation_models import QualityReport, TranslationValidator

    original_validate = TranslationValidator.validate

    def validate(self, source: str, translated: str, target_language: str = "中文"):
        report = original_validate(self, source, translated, target_language)
        if not looks_like_model_refusal(translated):
            return report
        reasons = tuple(
            _append_reason(
                getattr(report, "reasons", ()) or (),
                "模型返回安全拒答模板，禁止作为医学译文发布",
            )
        )
        return QualityReport(False, 0.0, reasons)

    validate._phoenix_refusal_validator = True
    TranslationValidator.validate = validate

    from .office_translation import OfficeDocumentTranslator

    original_audit = OfficeDocumentTranslator._decision_audit

    def decision_audit(segment, decision) -> dict:
        row = original_audit(segment, decision)
        candidate = str(row.get("translated", "") or "").strip()
        refused = looks_like_model_refusal(candidate)
        approved = (
            bool(row.get("quality_ok", False))
            and not bool(row.get("needs_review", True))
            and bool(candidate)
            and not refused
        )
        if approved:
            row["publication_approved"] = True
            return row

        # Preserve the rejected candidate for diagnostics/learning but never put
        # it into the formal Office deliverable.
        if candidate and candidate != segment.source:
            row["rejected_candidate"] = candidate[:4000]
        if refused:
            row["refused_output"] = candidate[:1200]

        row["translated"] = segment.source
        backend = str(row.get("backend", "") or "unknown")
        guard_suffix = (
            "refusal_guard_preserve_source"
            if refused
            else "quality_guard_preserve_source"
        )
        if guard_suffix not in backend:
            row["backend"] = f"{backend}|{guard_suffix}"
        row["quality_ok"] = False
        if refused:
            row["quality_score"] = 0.0
        row["reasons"] = _append_reason(
            row.get("reasons") or (),
            (
                "模型返回安全拒答模板；正式成品已自动保留英文原文"
                if refused
                else "译文未通过最终质量门；正式成品已自动保留原文"
            ),
        )
        row["needs_review"] = True
        row["publication_approved"] = False
        return row

    decision_audit._phoenix_office_publication_guard = True
    OfficeDocumentTranslator._decision_audit = staticmethod(decision_audit)
