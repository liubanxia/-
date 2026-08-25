from __future__ import annotations

"""Last-mile publication contract for the production translation runtime.

Context and model-routing installers may patch per-page/per-segment hooks, but
no installer after the stability core is allowed to wrap PDFTranslator.translate_book.
Keeping one deterministic publication boundary prevents resume, validation and
atomic-publish behavior from depending on installer order.
"""

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import translation_stability_core as stability
    from .translation_models import MultiModelTranslationEngine
    from .translator import PDFTranslator

    PDFTranslator.translate_book = stability._stable_translate_book
    PDFTranslator._phoenix_translation_wrapper_depth = 1
    PDFTranslator._phoenix_final_publication_contract_v2 = True

    # Historical installers can leave class-level fingerprints even after their
    # public wrappers have been collapsed. In the normal production path these
    # experimental cascades are not active, so stale markers must not make the
    # runtime claim that they are. If the developer explicitly enables the
    # experimental cascade later in bootstrap, its installer can set them again.
    for marker in (
        "_phoenix_translation_cascade_v2_installed",
        "_phoenix_hybrid_translation_policy_installed",
    ):
        if hasattr(MultiModelTranslationEngine, marker):
            delattr(MultiModelTranslationEngine, marker)

    _INSTALLED = True
