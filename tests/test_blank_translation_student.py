from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge import translation_blank_student as student


class BlankTranslationStudentTest(unittest.TestCase):
    def test_seed_is_intentionally_tiny_and_capacity_is_expandable(self):
        seed = student.parameter_count("seed")
        small = student.parameter_count("small")
        medium = student.parameter_count("medium")

        self.assertEqual(seed, 52883)
        self.assertLess(seed, 100_000)
        self.assertLess(seed * 4, 512 * 1024)
        self.assertLess(seed, small)
        self.assertLess(small, medium)

    def test_student_can_never_self_promote_to_production(self):
        self.assertFalse(student.PRODUCTION_ELIGIBLE)
        self.assertFalse(student.expert_admission_allowed())

    def test_observation_deduplicates_but_keeps_exposure_count(self):
        with tempfile.TemporaryDirectory() as directory:
            store = student.BlankStudentStore(Path(directory))
            first = store.observe(
                "No evidence of acute intracranial hemorrhage.",
                "未见急性颅内出血证据。",
                "中文",
                teacher_backend="qwen_local_medical_model3|quality_final",
                quality_score=0.95,
            )
            second = store.observe(
                "No evidence of acute intracranial hemorrhage.",
                "未见急性颅内出血证据。",
                "中文",
                teacher_backend="qwen35_medical_translation_translation_fallback_1",
                quality_score=0.98,
            )
            stats = store.stats("seed")

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(stats.samples, 1)
            self.assertEqual(stats.exposures, 2)
            self.assertEqual(stats.trained_samples, 0)

    def test_long_examples_are_kept_for_later_capacity_growth(self):
        with tempfile.TemporaryDirectory() as directory:
            store = student.BlankStudentStore(Path(directory))
            source = "A" * 220
            target = "中" * 70
            store.observe(
                source,
                target,
                "中文",
                teacher_backend="qwen_local_medical_model3|quality_final",
                quality_score=0.95,
            )

            self.assertEqual(store.stats("seed").samples, 1)
            self.assertEqual(store.training_rows(student.capacity_profile("seed")), [])
            self.assertEqual(
                len(store.training_rows(student.capacity_profile("medium"))),
                1,
            )

    def test_capacity_checkpoints_are_separate_but_corpus_is_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            store = student.BlankStudentStore(Path(directory))
            self.assertNotEqual(
                store.checkpoint_path("seed"),
                store.checkpoint_path("small"),
            )
            self.assertNotEqual(
                store.checkpoint_path("small"),
                store.checkpoint_path("medium"),
            )
            self.assertEqual(store.db_path.parent, store.checkpoint_path("seed").parent.parent)

    def test_training_is_document_bounded_and_not_sentence_inline(self):
        code = Path(student.__file__).read_text(encoding="utf-8")
        self.assertIn("_train_after_document", code)
        self.assertIn("PHOENIX_BLANK_STUDENT_TRAIN_SECONDS", code)
        self.assertIn("PDFTranslator.translate_book = pdf_book", code)
        self.assertIn("OfficeDocumentTranslator.translate_document = office_document", code)
        self.assertNotIn("cascade._translate =", code)
        self.assertNotIn("MultiModelTranslationEngine.translate =", code)

    def test_torch_is_lazy_and_collection_survives_without_training_runtime(self):
        code = Path(student.__file__).read_text(encoding="utf-8")
        top_level = code.split("def _build_model", 1)[0]
        self.assertNotIn("import torch", top_level)
        self.assertIn("torch_unavailable", code)
        self.assertIn("样本收集失败但正式翻译继续", code)
        self.assertIn("影子训练跳过但不影响译文", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
