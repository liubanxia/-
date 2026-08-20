from __future__ import annotations

import unittest

from phoenix_knowledge.translation_models import TranslationValidator


class TranslationShortChineseTests(unittest.TestCase):
    def setUp(self):
        self.validator = TranslationValidator()

    def test_concise_medical_chinese_is_not_rejected_as_untranslated(self):
        cases = (
            ("The lesion lies below the diaphragm.", "病灶位于膈下。"),
            ("Current CT examination demonstrates no pleural effusion.", "CT未见胸腔积液。"),
            ("A small lesion is present in the right kidney.", "右肾见小病灶。"),
        )
        for source, translated in cases:
            with self.subTest(source=source):
                report = self.validator.validate(source, translated, "中文")
                self.assertTrue(report.ok, report)

    def test_short_output_does_not_bypass_semantic_or_numeric_safety(self):
        dangerous = (
            ("Current CT examination demonstrates no pleural effusion.", "CT见胸腔积液。"),
            ("A small lesion is present in the right kidney.", "左肾见小病灶。"),
            ("The lesion measures 12 mm on CT.", "CT示病灶2 mm。"),
        )
        for source, translated in dangerous:
            with self.subTest(source=source):
                report = self.validator.validate(source, translated, "中文")
                self.assertFalse(report.ok, report)

    def test_english_or_too_short_output_is_still_rejected(self):
        cases = (
            ("The lesion lies below the diaphragm.", "The lesion lies below the diaphragm."),
            ("Current CT examination demonstrates no pleural effusion.", "正常。"),
        )
        for source, translated in cases:
            with self.subTest(source=source):
                report = self.validator.validate(source, translated, "中文")
                self.assertFalse(report.ok, report)


if __name__ == "__main__":
    unittest.main()
