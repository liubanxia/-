from __future__ import annotations

import unittest

from phoenix_knowledge.translation_model3_audit_acceleration import (
    _apply_audit_payload,
    _parse_audit_payload,
)


class Model3AuditAccelerationTest(unittest.TestCase):
    def test_pass_keeps_translation_without_rewrite(self):
        payload = _parse_audit_payload('{"status":"PASS"}')
        text, mode, edits = _apply_audit_payload("未见急性颅内出血证据。", payload)
        self.assertEqual(text, "未见急性颅内出血证据。")
        self.assertEqual(mode, "pass")
        self.assertEqual(edits, 0)

    def test_patch_applies_exact_unique_medical_correction(self):
        payload = _parse_audit_payload(
            '{"status":"PATCH","edits":[{"old":"可见急性颅内出血",'
            '"new":"未见急性颅内出血证据"}]}'
        )
        text, mode, edits = _apply_audit_payload(
            "CT示可见急性颅内出血。",
            payload,
        )
        self.assertEqual(text, "CT示未见急性颅内出血证据。")
        self.assertEqual(mode, "patch")
        self.assertEqual(edits, 1)

    def test_patch_rejects_ambiguous_replacement(self):
        payload = {
            "status": "PATCH",
            "edits": [{"old": "病灶", "new": "结节"}],
        }
        with self.assertRaises(ValueError):
            _apply_audit_payload("病灶旁另见病灶。", payload)

    def test_full_mode_remains_available_for_complex_errors(self):
        payload = {
            "status": "FULL",
            "final_text": "DWI示弥散受限，ADC值降低。",
        }
        text, mode, edits = _apply_audit_payload("旧译文", payload)
        self.assertEqual(text, "DWI示弥散受限，ADC值降低。")
        self.assertEqual(mode, "full")
        self.assertEqual(edits, 1)

    def test_markdown_wrapped_json_is_tolerated(self):
        payload = _parse_audit_payload('```json\n{"status":"PASS"}\n```')
        self.assertEqual(payload["status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
