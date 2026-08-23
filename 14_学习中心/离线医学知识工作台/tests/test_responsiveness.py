from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_download import GROUPS
from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.llm import LocalLLM
from phoenix_knowledge.retrieval import EmbeddingEngine


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


def _ready_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    # READY intentionally requires a non-empty real-weight-shaped file. Tests
    # must model the current product contract instead of reviving the old
    # folder-only readiness rule.
    (path / "model.safetensors").write_bytes(b"test-weight")


class _VectorDB:
    def __init__(self):
        self.calls = 0
        self.rows = []
        for chunk_id, values in (
            (11, [1.0, 0.0, 0.0]),
            (12, [0.0, 1.0, 0.0]),
            (13, [0.0, 0.0, 1.0]),
        ):
            vector = np.asarray(values, dtype=np.float32)
            self.rows.append(
                {
                    "chunk_id": chunk_id,
                    "dim": int(vector.size),
                    "vector": vector.tobytes(),
                }
            )

    def iter_embeddings(self, model_name):
        self.calls += 1
        return iter(self.rows)


class ResponsivenessTests(unittest.TestCase):
    def test_all_generator_requests_route_to_smart2(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            deep = paths.model_root / "Qwen3.5-4B"
            _ready_model(deep)

            llm = LocalLLM(paths)
            self.assertEqual(llm.selected_model("fast")[0], "Qwen3.5-4B")
            self.assertEqual(llm.selected_model("deep")[0], "Qwen3.5-4B")

    def test_fast_profile_falls_back_to_existing_4b(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            deep = paths.model_root / "Qwen3.5-4B"
            _ready_model(deep)

            llm = LocalLLM(paths)
            self.assertEqual(llm.selected_model("fast")[0], "Qwen3.5-4B")

    def test_vector_index_is_materialized_once_and_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            db = _VectorDB()
            engine = EmbeddingEngine(db, paths)

            ids1, matrix1 = engine._load_vector_index()
            ids2, matrix2 = engine._load_vector_index()

            self.assertEqual(db.calls, 1)
            self.assertIs(ids1, ids2)
            self.assertIs(matrix1, matrix2)
            self.assertEqual(matrix1.shape, (3, 3))
            self.assertEqual(ids1.tolist(), [11, 12, 13])

    def test_hospital_group_includes_fast_qa_and_formal_translation_models(self):
        self.assertIn("generator_fast", GROUPS["hospital_recommended"])
        self.assertIn("generator", GROUPS["hospital_recommended"])
        self.assertEqual(GROUPS["deep_quality"], ["generator"])


if __name__ == "__main__":
    unittest.main()
