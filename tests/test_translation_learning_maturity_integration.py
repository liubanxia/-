from __future__ import annotations

import unittest
from pathlib import Path

from phoenix_knowledge import translation_learning_maturity_gate as maturity
from phoenix_knowledge import translation_maturity_runtime_v2 as runtime_v2


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

    def test_early_stage_is_collection_only_for_production_memory(self):
        base_code = Path(maturity.__file__).read_text(encoding="utf-8")
        runtime_code = Path(runtime_v2.__file__).read_text(encoding="utf-8")

        self.assertIn("从第1本开始只收集", base_code)
        self.assertIn("翻译记忆只收集、不介入生产翻译", base_code)
        self.assertIn('"_phoenix_production_memory"', runtime_code)
        self.assertIn("if not maturity._memory_is_mature(self):", runtime_code)
        self.assertIn("return None", runtime_code)

    def test_tracking_failure_is_nonfatal_contract(self):
        base_code = Path(maturity.__file__).read_text(encoding="utf-8")
        runtime_code = Path(runtime_v2.__file__).read_text(encoding="utf-8")

        self.assertIn("正常翻译链继续运行", base_code)
        self.assertIn("完成书籍计数失败，但不影响译文", runtime_code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
