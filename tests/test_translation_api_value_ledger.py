from __future__ import annotations

from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phoenix_knowledge.translation_api_value_ledger import (
    APIValueLedger,
    _classify_errors,
    _is_api_backend,
)


class APIValueLedgerTest(unittest.TestCase):
    def test_api_backend_detection_does_not_count_memory_reuse(self):
        self.assertTrue(_is_api_backend("qwen35_medical_translation_model3_failed_translation_fallback_1"))
        self.assertFalse(_is_api_backend("translation_memory_exact:qwen35_medical_translation"))
        self.assertFalse(_is_api_backend("emergency_onnx_cpu"))

    def test_error_categories_keep_medical_failure_reason(self):
        labels = _classify_errors(
            "No evidence of a 12 mm pulmonary nodule.",
            "可见肺结节。",
            "未见12 mm肺结节证据。",
            ("数字/单位/正负号未完整保留(0%)",),
        )
        self.assertIn("number_unit", labels)
        self.assertIn("negation", labels)

    def test_successful_api_answer_is_candidate_not_expert(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = APIValueLedger(Path(tmp) / "api_value.sqlite3")
            local = SimpleNamespace(
                backend="qwen_local_medical_model3|quality_final_v2",
                text="可见急性颅内出血。",
                quality=SimpleNamespace(
                    ok=False,
                    score=0.55,
                    reasons=("医学语义终审未通过",),
                ),
            )
            api = SimpleNamespace(
                backend="qwen35_medical_translation_model3_failed_translation_fallback_1",
                text="未见急性颅内出血证据。",
                quality=SimpleNamespace(ok=True, score=0.95, reasons=()),
            )
            ledger.record_attempt(
                source="No evidence of acute intracranial hemorrhage.",
                target="中文",
                local_attempt=local,
                api_attempt=api,
            )
            stats = ledger.stats()
            self.assertEqual(stats.calls, 1)
            self.assertEqual(stats.accepted_calls, 1)
            self.assertEqual(stats.reusable_assets, 1)
            with closing(sqlite3.connect(str(ledger.path))) as db:
                row = db.execute(
                    "SELECT training_status, reviewed, expert_eligible "
                    "FROM api_learning_assets"
                ).fetchone()
            self.assertEqual(row, ("candidate_only", 0, 0))

    def test_ledger_never_calls_api_for_analysis(self):
        import phoenix_knowledge.translation_api_value_ledger as module

        code = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".llm.generate(", code)
        self.assertNotIn("retry_translation(", code)
        self.assertIn("NEVER makes an extra API request", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
