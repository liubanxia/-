from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest


class RuntimeBootstrapIsolationTest(unittest.TestCase):
    def _run(self, script: str):
        env = dict(os.environ)
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(completed.stdout or "") + "\n" + (completed.stderr or ""),
        )
        return completed

    def test_low_level_import_does_not_install_production_runtime(self):
        completed = self._run(
            r'''
            import tempfile
            from pathlib import Path

            import phoenix_knowledge
            assert not phoenix_knowledge.runtime_bootstrapped()

            from phoenix_knowledge.translation_survival_memory import TranslationMemory
            assert not phoenix_knowledge.runtime_bootstrapped()

            with tempfile.TemporaryDirectory() as td:
                memory = TranslationMemory(Path(td) / "memory.sqlite3")
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
                assert hit is not None
                assert hit.translation == "未见急性颅内出血证据。"
            print("LOW_LEVEL_IMPORT_ISOLATION=PASS")
            '''
        )
        self.assertIn("LOW_LEVEL_IMPORT_ISOLATION=PASS", completed.stdout)

    def test_production_bootstrap_is_idempotent_and_maturity_is_instance_scoped(self):
        completed = self._run(
            r'''
            import tempfile
            from pathlib import Path

            import phoenix_knowledge
            from phoenix_knowledge.translation_survival_memory import TranslationMemory

            first = phoenix_knowledge.bootstrap_runtime()
            second = phoenix_knowledge.bootstrap_runtime()
            assert first == second
            assert phoenix_knowledge.runtime_bootstrapped()
            assert "translation_maturity_runtime_v2" in first
            assert "translation_api_value_runtime_v2" in first
            assert "translation_learning_maturity_gate" not in first
            assert "translation_api_value_ledger" not in first

            with tempfile.TemporaryDirectory() as td:
                memory = TranslationMemory(Path(td) / "memory.sqlite3")
                memory.store(
                    "No evidence of acute intracranial hemorrhage.",
                    "未见急性颅内出血证据。",
                    "中文",
                    backend="api_teacher",
                    quality_score=1.0,
                )

                # Maintenance/test memory remains a normal data structure even
                # inside a bootstrapped production process.
                hit = memory.lookup_exact(
                    "no evidence of acute intracranial hemorrhage",
                    "中文",
                )
                assert hit is not None

                # Only the engine-owned production instance is maturity-gated.
                memory._phoenix_production_memory = True
                assert memory.lookup_exact(
                    "no evidence of acute intracranial hemorrhage",
                    "中文",
                ) is None

            print("PRODUCTION_SCOPE_CONTRACT=PASS")
            '''
        )
        self.assertIn("PRODUCTION_SCOPE_CONTRACT=PASS", completed.stdout)

    def test_partial_bootstrap_failure_latches_and_refuses_retry(self):
        completed = self._run(
            r'''
            import phoenix_knowledge.runtime_bootstrap as runtime
            import phoenix_knowledge.translation_recovery as recovery

            calls = {"count": 0}

            def fail_install():
                calls["count"] += 1
                raise RuntimeError("simulated installer failure")

            recovery.install = fail_install

            try:
                runtime.bootstrap_runtime()
            except runtime.RuntimeBootstrapError as exc:
                first = str(exc)
            else:
                raise AssertionError("first bootstrap unexpectedly succeeded")

            assert calls["count"] == 1
            assert runtime.runtime_bootstrap_failure() is not None
            assert "simulated installer failure" in runtime.runtime_bootstrap_failure()
            assert not runtime.runtime_bootstrapped()

            try:
                runtime.bootstrap_runtime()
            except runtime.RuntimeBootstrapError as exc:
                second = str(exc)
            else:
                raise AssertionError("second bootstrap unexpectedly succeeded")

            # The failed installer must never be invoked a second time in the
            # partially-mutated process.
            assert calls["count"] == 1
            assert "禁止重试" in second
            assert first != second
            print("BOOTSTRAP_FAILURE_LATCH=PASS")
            '''
        )
        self.assertIn("BOOTSTRAP_FAILURE_LATCH=PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
