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
    """Enforce the last publication gate for formal medical translation.

    Shared validation rejects model refusal templates. Office delivery goes one
    step further: only quality-approved, non-review output may replace source
    text. Rejected candidates remain in audit JSON but never enter PPTX/DOCX.
    If a document has translatable text and *no* segment passes this gate, the
    newly built Office file is removed before the method returns and the task is
    left failed/checkpointed instead of masquerading as a completed translation.
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

    from . import office_translation as office

    OfficeDocumentTranslator = office.OfficeDocumentTranslator
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

    original_translate_document = OfficeDocumentTranslator.translate_document

    def translate_document(self, source_path, *args, **kwargs):
        result = original_translate_document(self, source_path, *args, **kwargs)
        if bool(getattr(result, "paused", False)):
            return result

        try:
            source = office.Path(source_path).resolve()
            target = str(kwargs.get("target_language", "中文") or "中文")
            digest = office._sha256_file(source)
            root, units_root, _previews, checkpoint, final_output = self._task_paths(
                source,
                digest,
                target,
            )

            audited = 0
            approved = 0
            for unit_file in sorted(units_root.glob("*.json")):
                payload = office._read_json(unit_file)
                for row in payload.get("translations") or ():
                    if not isinstance(row, dict):
                        continue
                    audited += 1
                    if bool(row.get("publication_approved", False)):
                        approved += 1

            # Zero audited rows means the document simply had no translatable
            # text and is allowed to pass through unchanged. Any translatable
            # document with zero approved translations is a hard failure.
            if audited > 0 and approved == 0:
                office.Path(final_output).unlink(missing_ok=True)
                office.Path(getattr(result, "output_path", final_output)).unlink(
                    missing_ok=True
                )
                state = office._read_json(checkpoint)
                state.update(
                    {
                        "status": "failed",
                        "review_required": True,
                        "review_segments": audited,
                        "error": "Office正式发布门：没有任何片段通过质量审核，已阻断成品输出。",
                    }
                )
                office._write_json(checkpoint, state)
                raise office.OfficeTranslationError(
                    "Office医学翻译全部未通过最终质量门；已保留checkpoint/audit，"
                    "未发布译本。"
                )
        except office.OfficeTranslationError:
            raise
        except Exception as exc:
            # A failure in the safety audit itself must fail closed, not silently
            # publish an unverifiable formal file.
            try:
                office.Path(getattr(result, "output_path", "")).unlink(
                    missing_ok=True
                )
            except Exception:
                pass
            raise office.OfficeTranslationError(
                "Office正式发布安全复核失败，已阻断成品输出："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return result

    translate_document._phoenix_office_all_failed_guard = True
    OfficeDocumentTranslator.translate_document = translate_document
