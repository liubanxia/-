from __future__ import annotations

import unittest
from pathlib import Path


class TranslationStoragePolicyFileTests(unittest.TestCase):
    def test_policy_documents_single_pdf_default(self):
        root = Path(__file__).resolve().parents[1]
        policy = root / "TRANSLATION_STORAGE_POLICY.md"
        self.assertTrue(policy.is_file())
        text = policy.read_text(encoding="utf-8")
        self.assertIn("one complete PDF only", text)
        self.assertIn("Split volumes are opt-in", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
