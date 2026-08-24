from __future__ import annotations

import unittest

from phoenix_knowledge import medical_terminology_core as core
from phoenix_knowledge import medical_acronyms
from phoenix_knowledge import translation_dual_route_release as contextual


class MedicalTerminologyCoreTest(unittest.TestCase):
    def test_core_is_large_enough_for_offline_medical_translation(self):
        stats = core.core_stats()
        self.assertGreaterEqual(stats["unique_abbreviations"], 500)
        self.assertGreaterEqual(stats["abbreviation_senses"], 520)
        self.assertGreaterEqual(stats["phrase_aliases"], 250)
        self.assertGreaterEqual(stats["total_senses_and_aliases"], 780)

    def test_requested_high_frequency_terms_are_in_core(self):
        text = (
            "DWI shows restricted diffusion with low ADC. "
            "HbA1c, eGFR, pH and SpO2 were recorded. "
            "LGE and ECV were assessed."
        )
        terms = core.terms_for_text(text)
        for expected in ("DWI", "ADC", "HbA1c", "eGFR", "pH", "SpO2", "LGE", "ECV"):
            self.assertIn(expected, terms)

        prompt = core.prompt_for_text(text)
        self.assertIn("弥散加权成像", prompt)
        self.assertIn("表观弥散系数", prompt)
        self.assertIn("糖化血红蛋白A1c", prompt)
        self.assertIn("估算肾小球滤过率", prompt)
        self.assertIn("外周血氧饱和度", prompt)
        self.assertIn("钆延迟强化", prompt)
        self.assertIn("细胞外容积分数", prompt)

    def test_ambiguous_abbreviations_keep_multiple_senses(self):
        seed = core.acronym_seed()
        self.assertGreaterEqual(len(seed["ADC"]), 2)
        self.assertIn(("apparent diffusion coefficient", "表观弥散系数"), seed["ADC"])
        self.assertIn(("adenocarcinoma", "腺癌"), seed["ADC"])
        self.assertGreaterEqual(len(seed["PE"]), 2)
        self.assertGreaterEqual(len(seed["MR"]), 2)

    def test_core_extends_existing_acronym_resolver(self):
        self.assertIn("LGE", medical_acronyms.RADIOLOGY_SEED)
        self.assertIn(
            ("late gadolinium enhancement", "钆延迟强化"),
            medical_acronyms.RADIOLOGY_SEED["LGE"],
        )

    def test_contextual_chain_receives_core_terms_and_translations(self):
        holder = type("Holder", (), {})()
        holder._phoenix_previous_source = ""
        holder._phoenix_previous_translation = ""
        holder._phoenix_terms = ()

        token = contextual._push(
            holder,
            "LGE with elevated ECV and reduced eGFR.",
            "PDF第6页",
        )
        try:
            value = contextual._context()
            self.assertIn("LGE", value)
            self.assertIn("钆延迟强化", value)
            self.assertIn("ECV", value)
            self.assertIn("细胞外容积分数", value)
            self.assertIn("eGFR", value)
            self.assertIn("估算肾小球滤过率", value)
        finally:
            contextual._CTX.reset(token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
