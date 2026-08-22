from __future__ import annotations

from dataclasses import dataclass

from .translation_integration import build_translation_runtime


@dataclass(frozen=True)
class TranslationRuntimeAdapter:
    """Runtime adapter for translation execution routing.

    Keeps policy selection isolated before wiring into PDFTranslator.
    """

    def prepare(
        self,
        source_text: str,
        page_number: int,
        requested_level: str = "smart1",
    ):
        return build_translation_runtime(
            source_text=source_text,
            page_number=page_number,
            requested_level=requested_level,
        )


def prepare_translation_runtime(
    source_text: str,
    page_number: int,
    requested_level: str = "smart1",
):
    return TranslationRuntimeAdapter().prepare(
        source_text=source_text,
        page_number=page_number,
        requested_level=requested_level,
    )
