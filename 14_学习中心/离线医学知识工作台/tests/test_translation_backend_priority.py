from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.translation_models import MultiModelTranslationEngine


class _LLM:
    def available(self, profile=None):
        return True

    def backend(self, profile=None):
        return "transformers_local"

    def generate(self, prompt, max_new_tokens=1200, *, profile=None):
        return "智能模型译文"


class _Backend:
    def __init__(self, name: str):
        self.name = name

    def available(self):
        return True

    def translate(self, text: str):
        return f"{self.name}:{text}"

    def unload(self):
        pass


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


class TranslationBackendPriorityTests(unittest.TestCase):
    def _engine(self, root: Path):
        engine = MultiModelTranslationEngine(_paths(root), _LLM())
        engine.marian = _Backend("marian_en_zh")
        engine.nllb = _Backend("nllb_600m_en_zh")
        return engine

    def test_smart1_uses_dedicated_translation_before_qwen(self):
        with tempfile.TemporaryDirectory() as td:
            engine = self._engine(Path(td))
            names = [
                backend.name
                for backend in engine.active_backends("中文", "smart1")
            ]
            self.assertEqual(
                names,
                [
                    "marian_en_zh",
                    "nllb_600m_en_zh",
                    "qwen35_medical_translation",
                ],
            )

    def test_smart2_is_explicit_quality_first_route(self):
        with tempfile.TemporaryDirectory() as td:
            engine = self._engine(Path(td))
            names = [
                backend.name
                for backend in engine.active_backends("中文", "smart2")
            ]
            self.assertEqual(
                names,
                [
                    "qwen35_medical_translation",
                    "marian_en_zh",
                    "nllb_600m_en_zh",
                ],
            )

    def test_visible_backend_order_matches_default_whole_book_route(self):
        with tempfile.TemporaryDirectory() as td:
            engine = self._engine(Path(td))
            self.assertEqual(
                engine.available_backends(),
                [
                    "marian_en_zh",
                    "nllb_600m_en_zh",
                    "qwen35_medical_translation",
                ],
            )


if __name__ == "__main__":
    unittest.main()
