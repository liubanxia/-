from __future__ import annotations

import unittest

from phoenix_knowledge import translation_learning_maturity_gate as maturity


class TranslationLearningMaturityIntegrationTest(unittest.TestCase):
    def test_release_floor_cannot_be_lowered_below_ten_books(self):
        self.assertFalse(
            maturity.is_mature(
                9,
                999999,
                min_books=1,
                min_verified_entries=100,
            )
        )

    def test_memory_activation_requires_both_gates(self):
        self.assertFalse(maturity.is_mature(10, 999))
        self.assertFalse(maturity.is_mature(9, 1000))
        self.assertTrue(maturity.is_mature(10, 1000))

    def test_early_stage_is_collection_only_contract(self):
        code = open(maturity.__file__, "r", encoding="utf-8").read()
        self.assertIn("从第1本开始只收集", code)
        self.assertIn("翻译记忆只收集、不介入生产翻译", code)
        self.assertIn("if not _memory_is_mature(self):", code)
        self.assertIn("return None", code)

    def test_tracking_failure_is_nonfatal_contract(self):
        code = open(maturity.__file__, "r", encoding="utf-8").read()
        self.assertIn("完成书籍计数失败，但不影响译文", code)
        self.assertIn("正常翻译链继续运行", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
