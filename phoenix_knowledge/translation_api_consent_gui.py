from __future__ import annotations

import os

from PySide6.QtWidgets import QDialog


_INSTALLED = False
_TRANSLATION_API_FLAG = "PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"
_KNOWLEDGE_REMOTE_FLAG = "PHOENIX_KNOWLEDGE_ALLOW_REMOTE"


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def install(gui_module) -> None:
    """Keep explicit cloud consent consistent across knowledge and translation.

    The compute dialog historically enabled PHOENIX_KNOWLEDGE_ALLOW_REMOTE but
    formal medical translation additionally required
    PHOENIX_TRANSLATION_ALLOW_API_FALLBACK.  That split made a provider appear
    selected in the GUI while Smart2 translation still reported NOT READY.

    We only synchronize after the dialog has successfully accepted a Save.
    Invalid/incomplete forms leave both flags unchanged.  Switching back to
    local mode disables translation API fallback in the same accepted save.
    API keys remain session-only by design.
    """

    del gui_module
    global _INSTALLED
    if _INSTALLED:
        return

    from .compute_gui import ComputeSettingsDialog

    original_save = ComputeSettingsDialog.save

    def save(self) -> None:
        original_save(self)

        try:
            accepted = self.result() == int(QDialog.DialogCode.Accepted)
        except Exception:
            accepted = False
        if not accepted:
            return

        os.environ[_TRANSLATION_API_FLAG] = (
            "1" if _enabled(_KNOWLEDGE_REMOTE_FLAG) else "0"
        )

    ComputeSettingsDialog.save = save
    ComputeSettingsDialog.__phoenix_translation_api_consent_bridge__ = True
    _INSTALLED = True
