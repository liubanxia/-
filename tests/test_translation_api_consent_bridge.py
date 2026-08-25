from __future__ import annotations

import os
import unittest


class TranslationAPIConsentBridgeTests(unittest.TestCase):
    def setUp(self):
        self._old_knowledge = os.environ.get("PHOENIX_KNOWLEDGE_ALLOW_REMOTE")
        self._old_translation = os.environ.get("PHOENIX_TRANSLATION_ALLOW_API_FALLBACK")

    def tearDown(self):
        for name, value in (
            ("PHOENIX_KNOWLEDGE_ALLOW_REMOTE", self._old_knowledge),
            ("PHOENIX_TRANSLATION_ALLOW_API_FALLBACK", self._old_translation),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _install_with_fake_dialog(self, accepted: bool):
        from phoenix_knowledge import compute_gui
        from phoenix_knowledge import translation_api_consent_gui as bridge

        original_cls = compute_gui.ComputeSettingsDialog
        bridge._INSTALLED = False

        class FakeDialog:
            def save(self):
                self._accepted = accepted

            def result(self):
                from PySide6.QtWidgets import QDialog

                return int(
                    QDialog.DialogCode.Accepted
                    if getattr(self, "_accepted", False)
                    else QDialog.DialogCode.Rejected
                )

        try:
            compute_gui.ComputeSettingsDialog = FakeDialog
            bridge.install(None)
            return FakeDialog
        finally:
            compute_gui.ComputeSettingsDialog = original_cls

    def test_remote_save_enables_translation_fallback(self):
        FakeDialog = self._install_with_fake_dialog(True)
        os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "1"
        os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"] = "0"
        dialog = FakeDialog()
        dialog.save()
        self.assertEqual(os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"], "1")

    def test_local_save_disables_translation_fallback(self):
        FakeDialog = self._install_with_fake_dialog(True)
        os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "0"
        os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"] = "1"
        dialog = FakeDialog()
        dialog.save()
        self.assertEqual(os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"], "0")

    def test_rejected_save_does_not_change_translation_flag(self):
        FakeDialog = self._install_with_fake_dialog(False)
        os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "1"
        os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"] = "0"
        dialog = FakeDialog()
        dialog.save()
        self.assertEqual(os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"], "0")


if __name__ == "__main__":
    unittest.main()
