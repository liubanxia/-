from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.translation_models import (
    MultiModelTranslationEngine,
    TranslationValidator,
)


class DummyLLM:
    def available(self):
        return False


class FakeBackend:
    def __init__(self, name: str, text: str | None = None, error: Exception | None = None):
        self.name = name
        self.text = text
        self.error = error

    def available(self):
        return True

    def translate(self, text: str, *args):
        if self.error is not None:
            raise self.error
        return self.text or ""

    def unload(self):
        return None


class TranslationCascadeTest(unittest.TestCase):
    def _engine(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        paths = WorkbenchPaths(
            project_root=root,
            source_root=root / "pdf",
            runtime_root=root / "runtime",
            evidence_root=root / "evidence",
            model_root=root / "models",
            database=root / "runtime" / "knowledge.sqlite3",
            structure_root=root / "runtime" / "docling",
        ).ensure()
        engine = MultiModelTranslationEngine(paths, DummyLLM())
        return td, engine

    def test_validator_rejects_untranslated_english(self):
        validator = TranslationValidator()
        source = "A 12 mm pulmonary nodule with CT attenuation of -20 HU was identified."
        report = validator.validate(source, source, "中文")
        self.assertFalse(report.ok)
        self.assertTrue(any("未翻译" in reason or "英文残留" in reason for reason in report.reasons))

    def test_validator_preserves_numbers_and_units(self):
        validator = TranslationValidator()
        source = "The lesion measures 12 mm and the attenuation is -20 HU."
        good = "病灶大小为12 mm，CT衰减值为-20 HU。"
        bad = "病灶较小，CT衰减值较低。"
        self.assertTrue(validator.validate(source, good, "中文").ok)
        self.assertFalse(validator.validate(source, bad, "中文").ok)

    def test_failed_primary_falls_back_to_secondary(self):
        td, engine = self._engine()
        try:
            engine.marian = FakeBackend("marian_en_zh", error=RuntimeError("boom"))
            engine.nllb = FakeBackend(
                "nllb_600m_en_zh",
                "一个12 mm肺结节，CT衰减值为20 HU，建议结合影像表现评估。",
            )
            engine.qwen = FakeBackend("qwen35_medical_review", "不应被调用")
            result = engine.translate(
                "A 12 mm pulmonary nodule has CT attenuation of 20 HU and requires imaging assessment.",
                "中文",
            )
            self.assertEqual(result.backend, "nllb_600m_en_zh")
            self.assertFalse(result.needs_review)
        finally:
            td.cleanup()

    def test_low_quality_primary_automatically_uses_next_model(self):
        td, engine = self._engine()
        try:
            source = "A 15 mm pulmonary nodule demonstrates spiculation and pleural retraction on CT."
            engine.marian = FakeBackend("marian_en_zh", source)
            engine.nllb = FakeBackend(
                "nllb_600m_en_zh",
                "CT显示一个15 mm肺结节，可见毛刺征及胸膜牵拉征。",
            )
            engine.qwen = FakeBackend("qwen35_medical_review", "不应被调用")
            result = engine.translate(source, "中文")
            self.assertEqual(result.backend, "nllb_600m_en_zh")
            self.assertEqual(len(result.attempts), 2)
        finally:
            td.cleanup()

    def test_all_low_quality_returns_best_with_review_flag(self):
        td, engine = self._engine()
        try:
            source = "A 23 mm lesion was measured at 45 HU on CT and MRI correlation was advised."
            engine.marian = FakeBackend("marian_en_zh", "病灶。")
            engine.nllb = FakeBackend("nllb_600m_en_zh", "病变需要评估。")
            engine.qwen = FakeBackend("qwen35_medical_review", "CT提示病变，建议进一步检查。")
            result = engine.translate(source, "中文")
            self.assertTrue(result.needs_review)
            self.assertIn(result.backend, {"marian_en_zh", "nllb_600m_en_zh", "qwen35_medical_review"})
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
