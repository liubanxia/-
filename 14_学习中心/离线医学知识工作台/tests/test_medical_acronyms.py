from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phoenix_knowledge.medical_acronyms import (
    MedicalAcronymResolver,
    extract_acronyms,
)
from phoenix_knowledge.translation_models import (
    QwenMedicalTranslationBackend,
    TranslationValidator,
)


class _LLM:
    def __init__(self, response: str = "[]", *, available: bool = True):
        self.response = response
        self.is_available = available
        self.calls: list[tuple[str, int, str | None]] = []

    def available(self, profile=None):
        return self.is_available

    def generate(self, prompt, max_new_tokens=1200, *, profile=None):
        self.calls.append((prompt, max_new_tokens, profile))
        return self.response


def _paths(root: Path):
    model_root = root / "models"
    runtime_root = root / "runtime"
    model_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(model_root=model_root, runtime_root=runtime_root)


class MedicalAcronymTests(unittest.TestCase):
    def test_extracts_medical_tokens_but_ignores_uppercase_slide_words(self):
        self.assertEqual(
            extract_acronyms("IMAGING SHOWS DWI, ADC and SUVmax."),
            ("DWI", "ADC", "SUVmax"),
        )
        self.assertEqual(
            extract_acronyms("HbA1c, eGFR, HFrEF, SpO2 and pH"),
            ("HbA1c", "eGFR", "HFrEF", "SpO2", "pH"),
        )

    def test_unambiguous_radiology_seed_uses_zero_model_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            llm = _LLM()
            glossary = MedicalAcronymResolver(
                _paths(Path(temp)),
                llm,
            ).resolve([(1, "DWI, ADC, ECG and HbA1c")])

            self.assertEqual(glossary["DWI"]["chinese"], "弥散加权成像")
            self.assertEqual(glossary["ADC"]["chinese"], "表观弥散系数")
            self.assertEqual(glossary["ECG"]["chinese"], "心电图")
            self.assertEqual(glossary["HBA1C"]["chinese"], "糖化血红蛋白A1c")
            self.assertEqual(llm.calls, [])

    def test_ambiguous_term_is_resolved_once_from_nearby_deck_context(self):
        with tempfile.TemporaryDirectory() as temp:
            response = json.dumps(
                [
                    {
                        "acronym": "PE",
                        "english": "pulmonary embolism",
                        "chinese": "肺栓塞",
                    }
                ],
                ensure_ascii=False,
            )
            llm = _LLM(response)
            glossary = MedicalAcronymResolver(
                _paths(Path(temp)),
                llm,
            ).resolve(
                [
                    (
                        7,
                        "CTA demonstrates a pulmonary arterial filling defect, consistent with PE.",
                    )
                ]
            )

            self.assertEqual(glossary["PE"]["english"], "pulmonary embolism")
            self.assertEqual(glossary["PE"]["source"], "smart2_context_disambiguation")
            self.assertEqual(len(llm.calls), 1)
            prompt_text, _budget, profile = llm.calls[0]
            self.assertIn("第7页", prompt_text)
            self.assertIn("pleural effusion", prompt_text)
            self.assertEqual(profile, "translation")

    def test_optional_large_inventory_is_stream_filtered_to_current_deck(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / "inventory.csv"
            inventory.write_text(
                "SF,LF\nABC,airway breathing circulation\nXYZ,unrelated sense\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"PHOENIX_MEDICAL_ACRONYM_CSV": str(inventory)},
                clear=False,
            ):
                resolver = MedicalAcronymResolver(_paths(root), _LLM(available=False))
                candidates = resolver._external_candidates({"ABC"})

            self.assertEqual(candidates["ABC"], ["airway breathing circulation"])
            self.assertNotIn("XYZ", candidates)

    def test_batch_prompt_carries_only_acronyms_used_in_that_batch(self):
        backend = QwenMedicalTranslationBackend(_LLM())
        backend.set_document_glossary(
            {
                "DWI": {
                    "acronym": "DWI",
                    "english": "diffusion-weighted imaging",
                    "chinese": "弥散加权成像",
                },
                "ADC": {
                    "acronym": "ADC",
                    "english": "apparent diffusion coefficient",
                    "chinese": "表观弥散系数",
                },
            }
        )

        prompt = backend.glossary_prompt("Restricted diffusion on DWI.")

        self.assertIn("DWI = diffusion-weighted imaging = 弥散加权成像", prompt)
        self.assertNotIn("ADC", prompt)

    def test_pure_acronym_cannot_pass_as_an_unchanged_chinese_translation(self):
        validator = TranslationValidator()

        unchanged = validator.validate("DWI", "DWI", "中文")
        expanded = validator.validate("DWI", "弥散加权成像（DWI）", "中文")

        self.assertFalse(unchanged.ok)
        self.assertIn("独立医学缩写未给出中文释义", unchanged.reasons)
        self.assertTrue(expanded.ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
