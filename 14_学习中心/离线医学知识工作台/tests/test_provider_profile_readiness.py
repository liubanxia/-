from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.llm_safe import LocalLLM


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


class ProviderProfileReadinessTests(unittest.TestCase):
    def test_custom_remote_fast_and_deep_are_checked_separately(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "custom_openai",
            },
            clear=False,
        ):
            llm = LocalLLM(_paths(Path(temp)))
            llm.compute.select_provider(
                "custom_openai",
                base_url="http://127.0.0.1:8000/v1",
                fast_model="local-fast",
                deep_model="",
            )
            # select_provider falls back to the preset default only when one
            # exists. custom_openai has no default deep model, so Smart2 must
            # remain unavailable instead of inheriting Smart1.
            self.assertTrue(llm.available("fast"))
            self.assertFalse(llm.available("deep"))

    def test_openai_has_two_explicit_current_profiles(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
                "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
                "PHOENIX_KNOWLEDGE_PROVIDER": "openai",
                "PHOENIX_PROVIDER_OPENAI_API_KEY": "test-key",
            },
            clear=False,
        ):
            llm = LocalLLM(_paths(Path(temp)))
            llm.compute.select_provider("openai")
            self.assertEqual(
                llm.compute.remote_model("fast"),
                "gpt-5.6-luna",
            )
            self.assertEqual(
                llm.compute.remote_model("deep"),
                "gpt-5.6-sol",
            )
            self.assertTrue(llm.available("fast"))
            self.assertTrue(llm.available("deep"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
