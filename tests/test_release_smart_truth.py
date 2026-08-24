from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.llm_safe import LocalLLM
from phoenix_knowledge.release_runtime_hardening import (
    _native_generator_ready,
)


def _paths(root: Path) -> WorkbenchPaths:
    return WorkbenchPaths(
        project_root=root,
        source_root=root / "sources",
        runtime_root=root / "runtime",
        evidence_root=root / "evidence",
        model_root=root / "models",
        database=root / "runtime" / "knowledge.sqlite3",
        structure_root=root / "runtime" / "structure",
    ).ensure()


def _model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


class SmartTruthTests(unittest.TestCase):
    def test_single_smart2_model_is_ready_for_all_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            _model(paths.model_root / "Qwen3.5-4B")
            llm = LocalLLM(paths)
            self.assertEqual(
                llm.active_model_name("deep"),
                "Qwen3.5-4B",
            )
            with patch(
                "phoenix_knowledge.release_runtime_hardening.local_generation_runtime_ready",
                return_value=True,
            ):
                self.assertTrue(
                    _native_generator_ready(llm, "fast")
                )
                self.assertTrue(
                    _native_generator_ready(llm, "deep")
                )

    def test_deep_profile_reports_actual_fast_model_fallback(self):
        import phoenix_knowledge.release_gui_truth as truth

        class _Signal:
            def __init__(self):
                self.values = []

            def emit(self, value):
                self.values.append(value)

        class _Answer:
            mode = "grounded_generation"
            text = "结论 [S1]"

        class _LLM:
            def active_model_name(self, profile=None):
                return "Qwen3.5-2B"

        class _WB:
            llm = _LLM()

            def ask(self, query, **kwargs):
                return _Answer()

        class _Worker:
            def __init__(self):
                self.workbench = _WB()
                self.query = "测试"
                self.completed = _Signal()
                self.failed = _Signal()

        class _Window:
            def _status_text(self):
                return "status"

            def refresh_translation_models(self):
                return None

        class _Module:
            AskWorker = _Worker
            WorkbenchWindow = _Window

        truth._INSTALLED = False
        truth.install(_Module)
        worker = _Worker()
        with patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_DEEP_QA": "1",
                "PHOENIX_KNOWLEDGE_LLM_PROFILE": "deep",
            },
            clear=False,
        ):
            worker.run()
        self.assertFalse(worker.failed.values)
        self.assertIn(
            "Smart2 · Qwen3.5-2B",
            worker.completed.values[-1],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
