from __future__ import annotations

import unittest

from phoenix_knowledge import translation_dual_route_release as contextual
from phoenix_knowledge.translation_refusal_guard import looks_like_model_refusal


class _Holder:
    pass


class ContextualTranslationChainContractTest(unittest.TestCase):
    def tearDown(self):
        contextual._CTX.set({})

    def test_medical_term_extractor_keeps_acronyms_and_named_terms(self):
        terms = contextual._terms(
            "DWI demonstrates restricted diffusion with low ADC in acute cerebral infarction."
        )
        self.assertIn("DWI", terms)
        self.assertIn("ADC", terms)
        self.assertTrue(any("infarction" in value.lower() for value in terms))

    def test_context_links_previous_and_current_document_units(self):
        holder = _Holder()
        holder._phoenix_previous_source = "Cardiac MRI demonstrates late gadolinium enhancement."
        holder._phoenix_previous_translation = "心脏MRI显示钆延迟强化。"
        holder._phoenix_terms = ("LGE",)

        token = contextual._push(
            holder,
            "LGE is predominantly subepicardial.",
            "PDF第6页",
        )
        try:
            prompt_context = contextual._context()
            self.assertIn("上一单元英文", prompt_context)
            self.assertIn("上一单元已确定译文", prompt_context)
            self.assertIn("PDF第6页", prompt_context)
            self.assertIn("LGE", prompt_context)
        finally:
            contextual._CTX.reset(token)

    def test_refusal_template_remains_blocked(self):
        self.assertTrue(
            looks_like_model_refusal(
                "系统检测到您输入的内容可能涉及医疗或健康领域，因此我无法直接处理。"
            )
        )

    def test_api_learning_is_candidate_only_not_online_training(self):
        with open(contextual.__file__, "r", encoding="utf-8") as handle:
            code = handle.read()
        self.assertIn('"training_status": "candidate_only"', code)
        self.assertIn('"reviewed": False', code)
        self.assertNotIn("optimizer.step(", code)
        self.assertNotIn("backward()", code)

    def test_pipeline_contract_keeps_model2_conditional_and_model3_final(self):
        with open(contextual.__file__, "r", encoding="utf-8") as handle:
            code = handle.read()
        self.assertIn("if not m1_ok:", code)
        self.assertIn("hymt._run_model2", code)
        self.assertIn("backend.refine(source, base.text, target)", code)
        self.assertIn('return final, "quality_final_model3"', code)
        self.assertIn("_api_polish_local_draft", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
