from __future__ import annotations

from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.translation_survival_memory import (
    TranslationMemory,
    _safety_signature,
    deterministic_medical_translation,
    normalize_source,
)


class TranslationSurvivalMemoryTest(unittest.TestCase):
    def test_normalization_reuses_case_spacing_and_terminal_punctuation(self):
        self.assertEqual(
            normalize_source("  No evidence of acute intracranial hemorrhage. "),
            normalize_source("no evidence of acute intracranial hemorrhage"),
        )

    def test_exact_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            memory = TranslationMemory(Path(td) / "translation_memory.sqlite3")
            memory.store(
                "No evidence of acute intracranial hemorrhage.",
                "未见急性颅内出血证据。",
                "中文",
                backend="api_teacher",
                quality_score=1.0,
            )
            hit = memory.lookup_exact(
                "no evidence of acute intracranial hemorrhage",
                "中文",
            )
            self.assertIsNotNone(hit)
            self.assertEqual(hit.translation, "未见急性颅内出血证据。")

    def test_pending_queue_deduplicates_and_counts(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "translation_memory.sqlite3"
            memory = TranslationMemory(path)
            memory.enqueue_pending("Rare medical sentence.", "中文", "offline")
            memory.enqueue_pending("Rare medical sentence.", "中文", "offline again")

            with closing(sqlite3.connect(str(path))) as db:
                row = db.execute(
                    "SELECT attempts FROM pending_translation"
                ).fetchone()
            self.assertEqual(row[0], 2)

    def test_high_frequency_medical_sentence_is_zero_model(self):
        self.assertEqual(
            deterministic_medical_translation(
                "No evidence of acute intracranial hemorrhage."
            ),
            "未见急性颅内出血证据。",
        )
        self.assertEqual(
            deterministic_medical_translation("No pleural effusion or pneumothorax."),
            "未见胸腔积液或气胸。",
        )

    def test_slot_pattern_only_uses_known_medical_term(self):
        self.assertEqual(
            deterministic_medical_translation("No evidence of pulmonary embolism."),
            "未见肺栓塞证据。",
        )
        self.assertIsNone(
            deterministic_medical_translation(
                "No evidence of an invented and unresolved rare entity."
            )
        )

    def test_similarity_safety_signature_separates_negation_and_laterality(self):
        positive = "There is a 12 mm right renal lesion."
        negative = "There is no 12 mm right renal lesion."
        left = "There is a 12 mm left renal lesion."
        self.assertNotEqual(_safety_signature(positive), _safety_signature(negative))
        self.assertNotEqual(_safety_signature(positive), _safety_signature(left))

    def test_survival_layer_keeps_onnx_optional_and_cpu_only(self):
        import phoenix_knowledge.translation_survival_memory as module

        code = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn("CPUExecutionProvider", code)
        self.assertIn("Emergency-Translator-ONNX", code)
        self.assertIn("offline_pending_source_preserved", code)
        self.assertIn("相似句仅供模型3纠错", code)
        self.assertNotIn("optimizer.step(", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
