from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phoenix_knowledge.translation_api_value_ledger import APIValueLedger
from phoenix_knowledge.translation_blank_student import BlankStudentStore
from phoenix_knowledge.translation_learning_maturity_gate import TranslationLearningMaturity
from phoenix_knowledge.translation_survival_memory import TranslationMemory


class SQLiteLifecycleContractTest(unittest.TestCase):
    def test_all_transient_translation_databases_release_windows_file_handles(self):
        # TemporaryDirectory cleanup is the assertion: on Windows any leaked
        # sqlite3.Connection raises WinError 32 when this context exits.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            memory = TranslationMemory(root / "memory.sqlite3")
            memory.store(
                "No evidence of acute intracranial hemorrhage.",
                "未见急性颅内出血证据。",
                "中文",
                backend="test",
                quality_score=1.0,
            )
            self.assertIsNotNone(
                memory.lookup_exact(
                    "no evidence of acute intracranial hemorrhage",
                    "中文",
                )
            )

            maturity = TranslationLearningMaturity(root / "maturity.sqlite3")
            maturity.record_completed_book("hash", "book.pdf")
            self.assertEqual(maturity.stats().completed_books, 1)

            ledger = APIValueLedger(root / "api.sqlite3")
            local = SimpleNamespace(
                backend="model3",
                text="错误译文",
                quality=SimpleNamespace(ok=False, score=0.4, reasons=("测试",)),
            )
            api = SimpleNamespace(
                backend="api_teacher",
                text="正确译文",
                quality=SimpleNamespace(ok=True, score=0.95, reasons=()),
            )
            ledger.record_attempt(
                source="Example source.",
                target="中文",
                local_attempt=local,
                api_attempt=api,
            )
            self.assertEqual(ledger.stats().calls, 1)

            student = BlankStudentStore(root / "student")
            self.assertTrue(
                student.observe(
                    "Example source.",
                    "示例译文。",
                    "中文",
                    teacher_backend="api_teacher",
                    quality_score=0.95,
                )
            )
            self.assertEqual(student.stats().samples, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
