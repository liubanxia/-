from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.release_hardening import _prepare_hard_failed_pages
from phoenix_knowledge.retrieval import EmbeddingEngine
from phoenix_knowledge.translation_models import MultiModelTranslationEngine
from phoenix_knowledge.translator import PDFTranslator


class _UnavailableLLM:
    def available(self, *args, **kwargs):
        return False

    def unload(self):
        return None


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


class ReleaseLaunchHardeningTests(unittest.TestCase):
    def test_semantic_readiness_requires_runtime_and_full_vector_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            model = paths.model_root / "Qwen3-Embedding-0.6B"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")

            db = KnowledgeDB(paths.database)
            try:
                doc_id = db.upsert_document(
                    Path(temp) / "book.pdf",
                    "abc",
                    "book",
                    1,
                )
                chunk_ids = db.replace_page_chunks(
                    doc_id,
                    1,
                    ["first evidence", "second evidence"],
                )
                db.store_embeddings(
                    "Qwen3-Embedding-0.6B",
                    [(chunk_ids[0], 2, b"\x00" * 8)],
                )

                engine = EmbeddingEngine(db, paths)
                with patch(
                    "phoenix_knowledge.release_runtime_hardening._module_importable",
                    return_value=True,
                ):
                    state = engine.readiness()
                self.assertFalse(state["ready"])
                self.assertEqual(state["state"], "index_incomplete")
                self.assertEqual(state["vectors"], 1)
                self.assertEqual(state["missing"], 1)

                with patch(
                    "phoenix_knowledge.release_runtime_hardening._module_importable",
                    return_value=False,
                ):
                    state = engine.readiness()
                self.assertEqual(state["state"], "runtime_missing")
            finally:
                db.close()

    def test_commercial_release_filters_noncommercial_nllb_backend(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PHOENIX_COMMERCIAL_RELEASE": "1"},
            clear=False,
        ):
            engine = MultiModelTranslationEngine(
                _paths(Path(temp)),
                _UnavailableLLM(),
            )
            engine.qwen.available = lambda *args, **kwargs: False
            engine.marian.available = lambda: False
            engine.nllb.available = lambda: True

            self.assertNotIn(
                engine.nllb.name,
                engine.available_backends(),
            )
            active = engine.active_backends("中文", "smart1")
            self.assertFalse(
                any(
                    getattr(item, "name", "") == engine.nllb.name
                    for item in active
                )
            )

    def test_hard_failed_translation_page_is_forced_back_into_resume_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            source = root / "book.pdf"
            source.write_bytes(b"synthetic-source")
            translator = PDFTranslator(paths, _UnavailableLLM())

            from phoenix_knowledge.pdf_parser import sha256_file

            digest = sha256_file(source)
            _, pages_root, audit_root, _, _ = translator._book_paths(
                source.resolve(),
                digest,
                "中文",
            )
            page_file = pages_root / "000001.txt"
            page_file.write_text(
                "[自动翻译失败；已保留原文，待重试]\nsource",
                encoding="utf-8",
            )
            audit = {
                "page": 1,
                "warning_count": 1,
                "parts": [
                    {
                        "part": 1,
                        "backend": "failed_all",
                        "quality_ok": False,
                    }
                ],
            }
            (audit_root / "000001.json").write_text(
                json.dumps(audit, ensure_ascii=False),
                encoding="utf-8",
            )

            removed = _prepare_hard_failed_pages(
                translator,
                source,
                "中文",
            )
            self.assertEqual(removed, 1)
            self.assertFalse(page_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
