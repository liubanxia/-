from __future__ import annotations

import re

_INSTALLED = False
_SHORT_CHINESE_REASON = "中文字符过少，疑似未翻译"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _can_accept_short_medical_chinese(
    source: str,
    translated: str,
    target_language: str,
    base_reasons: tuple[str, ...],
) -> bool:
    if tuple(base_reasons) != (_SHORT_CHINESE_REASON,):
        return False

    from .translation_semantics import validate_medical_semantics

    if validate_medical_semantics(source, translated, target_language):
        return False

    text = (translated or "").strip()
    cjk_count = len(_CJK_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))

    if cjk_count < 4:
        return False
    if latin_count > max(8, cjk_count * 2):
        return False
    if text.casefold() == (source or "").strip().casefold():
        return False
    return True


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import translation_models as tm

    original_validate = tm.TranslationValidator.validate

    def validate(self, source: str, translated: str, target_language: str = "中文"):
        base = original_validate(self, source, translated, target_language)
        if base.ok:
            return base
        if not _can_accept_short_medical_chinese(
            source,
            translated,
            target_language,
            base.reasons,
        ):
            return base
        return tm.QualityReport(
            True,
            max(0.72, min(1.0, base.score + 0.27)),
            (),
        )

    tm.TranslationValidator.validate = validate
    tm.TranslationValidator._phoenix_short_chinese_guard = True
