from __future__ import annotations

import unittest
from unittest.mock import patch

from phoenix_knowledge.translation_chain_enforcement_v3 import _run_quality_chain
from phoenix_knowledge.translation_models import QualityReport, TranslationAttempt


class _Validator:
    def validate(self, source, translated, target_language="中文"):
        return QualityReport(True, 0.95, ())


class _Engine:
    def __init__(self):
        self.validator = _Validator()


class _Model3Refine:
    name = "qwen_local_medical_model3"

    def __init__(self, calls):
        self.calls = calls

    def refine(self, source, draft, target_language):
        self.calls.append("m3")
        return "模型3终审译文"


class _Model3Source:
    name = "qwen_local_medical_model3"

    def __init__(self, calls):
        self.calls = calls

    def _load(self):
        self.calls.append("m3-load")

    def _chat_prompt(self, system, user):
        return system + "\n" + user

    def _generate_prompt(self, prompt, draft, **kwargs):
        self.calls.append("m3-source")
        return "模型3源文直译"


class TranslationChainEnforcementV3Test(unittest.TestCase):
    def test_model2_runs_even_when_model1_is_good_and_model3_runs_after_it(self):
        from phoenix_knowledge import hymt_cascade_policy as hymt
        from phoenix_knowledge import translation_cascade_v2 as cascade
        from phoenix_knowledge import translation_dual_route_release as dual

        engine = _Engine()
        attempts = []
        errors = []
        calls = []

        m1 = TranslationAttempt(
            backend="model1_draft:test",
            text="模型1初稿",
            quality=QualityReport(True, 0.99, ()),
        )
        m2 = TranslationAttempt(
            backend="hymt15_1p8b_refine",
            text="模型2纠错",
            quality=QualityReport(True, 0.98, ()),
        )

        def run_m1(*args, **kwargs):
            calls.append("m1")
            attempts.append(m1)
            return m1

        def run_m2(*args, **kwargs):
            calls.append("m2")
            attempts.append(m2)
            return m2

        model3 = _Model3Refine(calls)
        with (
            patch.object(dual, "_model1", side_effect=run_m1),
            patch.object(hymt, "_run_model2", side_effect=run_m2),
            patch.object(cascade, "_model3_available", return_value=True),
            patch.object(cascade, "_model3", return_value=model3),
        ):
            result, stage = _run_quality_chain(
                engine,
                "English medical source",
                "中文",
                attempts,
                errors,
            )

        self.assertEqual(calls, ["m1", "m2", "m3"])
        self.assertEqual(stage, "quality_final_model3")
        self.assertTrue(result.backend.startswith("qwen_local_medical_model3"))
        self.assertEqual(errors, [])

    def test_model3_translates_source_when_model1_and_model2_have_no_draft(self):
        from phoenix_knowledge import hymt_cascade_policy as hymt
        from phoenix_knowledge import translation_cascade_v2 as cascade
        from phoenix_knowledge import translation_dual_route_release as dual

        engine = _Engine()
        attempts = []
        errors = []
        calls = []
        model3 = _Model3Source(calls)

        def no_m1(*args, **kwargs):
            calls.append("m1")
            return None

        def no_m2(*args, **kwargs):
            calls.append("m2")
            return None

        with (
            patch.object(dual, "_model1", side_effect=no_m1),
            patch.object(hymt, "_run_model2", side_effect=no_m2),
            patch.object(cascade, "_model3_available", return_value=True),
            patch.object(cascade, "_model3", return_value=model3),
        ):
            result, stage = _run_quality_chain(
                engine,
                "English medical source",
                "中文",
                attempts,
                errors,
            )

        self.assertEqual(calls, ["m1", "m2", "m3-load", "m3-source"])
        self.assertEqual(stage, "quality_final_model3_source")
        self.assertIn(":source", result.backend)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
