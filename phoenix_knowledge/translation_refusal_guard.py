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
    """Prevent model refusal templates from entering medical deliverables.

    The shared validator marks refusal text invalid for every translation path.
    Office delivery adds a second protection layer: the refusal is replaced by
    the original source text, marked review-required, and never cached as a
    valid translation. This keeps PPTX/DOCX usable for doctor correction while
    preventing safety-template prose from contaminating the translated file.
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

    TranslationValidator.validate = validate

    from .office_translation import OfficeDocumentTranslator

    original_audit = OfficeDocumentTranslator._decision_audit

    def decision_audit(segment, decision) -> dict:
        row = original_audit(segment, decision)
        translated = str(row.get("translated", "") or "").strip()
        if not looks_like_model_refusal(translated):
            return row

        row["refused_output"] = translated[:1200]
        row["translated"] = segment.source
        backend = str(row.get("backend", "") or "unknown")
        row["backend"] = f"{backend}|refusal_guard_preserve_source"
        row["quality_ok"] = False
        row["quality_score"] = 0.0
        row["reasons"] = _append_reason(
            row.get("reasons") or (),
            "模型返回安全拒答模板；已自动保留英文原文供医生修改",
        )
        row["needs_review"] = True
        return row

    OfficeDocumentTranslator._decision_audit = staticmethod(decision_audit)
