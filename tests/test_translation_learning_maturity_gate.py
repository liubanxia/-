from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.translation_learning_maturity_gate import (
    TranslationLearningMaturity,
    is_mature,
)
from phoenix_knowledge.translation_survival_memory import TranslationMemory


class TranslationLearningMaturityGateTest(unittest.TestCase):
    def test_nine_books_never_activate_memory(self):
        self.assertFalse(is_mature(9, 100000))

    def test_ten_books_without_enough_verified_rows_stays_dormant(self):
        self.assertFalse(is_mature(10, 999))

    def test_ten_books_and_enough_verified_rows_activate(self):
        self.assertTrue(is_mature(10, 1000))

    def test_same_book_hash_is_counted_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            TranslationMemory(path)
            gate = TranslationLearningMaturity(path)
            gate.record_completed_book("same-hash", "A.pdf")
            gate.record_completed_book("same-hash", "renamed.pdf")
            self.assertEqual(gate.stats().completed_books, 1)

    def test_verified_rows_are_counted_from_translation_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = TranslationMemory(path)
            gate = TranslationLearningMaturity(path)
            memory.store(
                "No evidence of acute intracranial hemorrhage.",
                "未见急性颅内出血证据。",
                "中文",
                backend="test",
                quality_score=1.0,
                verified_level=1,
            )
            self.assertEqual(gate.stats().verified_entries, 1)

    def test_gate_is_fail_closed_for_learning_not_for_translation(self):
        code = Path(__import__(
            "phoenix_knowledge.translation_learning_maturity_gate",
            fromlist=["x"],
        ).__file__).read_text(encoding="utf-8")
        self.assertIn("return False", code)
        self.assertIn("正常翻译链继续运行", code)
        self.assertIn("翻译记忆只收集、不介入生产翻译", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
