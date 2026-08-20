from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import runtime_preflight
from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.llm_safe import LocalLLM as SafeLocalLLM
from phoenix_knowledge.release_hardening import _runtime_module_available
from phoenix_knowledge.release_memory_hardening import (
    _unload_embeddings,
    _unload_llm,
)
from phoenix_knowledge.rich_export import MultiFormatExporter
from phoenix_knowledge.translation_models import MultiModelTranslationEngine


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


class _UnavailableLLM:
    def available(self, *args, **kwargs):
        return False

    def unload(self):
        return None


class ReleaseAuditV2Tests(unittest.TestCase):
    def test_optional_ai_dependencies_do_not_block_core_launch(self):
        fake_core = []
        fake_ai = ["sentence-transformers", "torch"]
        with patch.object(
            runtime_preflight,
            "_missing_group",
            side_effect=[fake_core, fake_ai],
        ):
            core, ai = runtime_preflight._missing_groups()
        self.assertEqual(core, fake_core)
        self.assertEqual(ai, fake_ai)

    def test_runtime_requirements_include_torch(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "requirements-runtime.txt"
        )
        text = path.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^torch\s*$")

    def test_semantic_runtime_ready_requires_real_import(self):
        with patch(
            "phoenix_knowledge.release_hardening.importlib.import_module",
            side_effect=ImportError("broken binary dependency"),
        ):
            self.assertFalse(
                _runtime_module_available("sentence_transformers")
            )

    def test_local_qwen_folder_is_not_ready_when_generation_runtime_is_broken(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            model = paths.model_root / "Qwen3.5-2B"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            llm = SafeLocalLLM(paths)
            self.assertEqual(llm.backend("fast"), "transformers_local")
            with patch(
                "phoenix_knowledge.release_runtime_hardening.local_generation_runtime_ready",
                return_value=False,
            ):
                self.assertFalse(llm.available("fast"))

    def test_seq2seq_translation_fallbacks_require_runtime_not_just_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = _paths(Path(temp))
            for folder in (
                "opus-mt-en-zh",
                "NLLB-200-distilled-600M",
            ):
                model = paths.model_root / folder
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}", encoding="utf-8")
                (model / "model.safetensors").write_bytes(b"weights")

            engine = MultiModelTranslationEngine(paths, _UnavailableLLM())
            with patch(
                "phoenix_knowledge.release_runtime_hardening.local_seq2seq_runtime_ready",
                return_value=False,
            ):
                names = engine.available_backends()
                active = engine.active_backends("中文", "smart1")
            self.assertNotIn(engine.marian.name, names)
            self.assertNotIn(engine.nllb.name, names)
            self.assertFalse(
                any(
                    getattr(item, "name", "")
                    in {engine.marian.name, engine.nllb.name}
                    for item in active
                )
            )

    def test_preflight_reports_capabilities_independently(self):
        class _LLM:
            def available(self, profile=None):
                return profile in {"fast", "deep"}

        fake = SimpleNamespace(llm=_LLM())
        flags = runtime_preflight._capability_flags(
            fake,
            {
                "semantic_ready": False,
                "generator_fast_ready": True,
                "generator_deep_ready": True,
                "translation_backends": ["qwen35_medical_translation"],
            },
        )
        self.assertFalse(flags["semantic"])
        self.assertTrue(flags["smart1"])
        self.assertTrue(flags["smart2"])
        self.assertTrue(flags["translation"])

    def test_rich_pdf_export_keeps_chinese_and_citations(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exporter = MultiFormatExporter(root / "out")
            long_text = (
                "肺结节影像学征象与鉴别诊断。[S12]\n" * 180
            )
            bundle = exporter.export_text(
                "# 上线验收\n\n" + long_text,
                title="上线验收",
            )
            self.assertTrue(bundle.pdf.is_file())
            doc = fitz.open(bundle.pdf)
            try:
                self.assertGreater(doc.page_count, 1)
                extracted = "\n".join(
                    page.get_text("text") for page in doc
                )
            finally:
                doc.close()
            self.assertIn("[S12]", extracted)
            self.assertIn("肺结节", extracted)

    def test_acceptance_forces_local_compute_by_default(self):
        env = {
            "PHOENIX_KNOWLEDGE_ACCELERATOR": "remote",
            "PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1",
            "PHOENIX_KNOWLEDGE_REMOTE_API_KEY": "secret",
        }
        with patch.dict(os.environ, env, clear=False):
            from real_acceptance import _force_local_acceptance

            _force_local_acceptance()
            self.assertEqual(
                os.environ.get("PHOENIX_KNOWLEDGE_ACCELERATOR"),
                "auto",
            )
            self.assertEqual(
                os.environ.get("PHOENIX_KNOWLEDGE_ALLOW_REMOTE"),
                "0",
            )
            self.assertNotIn(
                "PHOENIX_KNOWLEDGE_REMOTE_API_KEY",
                os.environ,
            )

    def test_release_memory_helpers_unload_resident_models(self):
        class _Embedding:
            def __init__(self):
                self.calls = 0

            def unload_model(self):
                self.calls += 1

        class _Retriever:
            def __init__(self):
                self.embeddings = _Embedding()

        class _LLM:
            def __init__(self):
                self.calls = 0

            def unload(self):
                self.calls += 1

        class _Workbench:
            def __init__(self):
                self.retriever = _Retriever()
                self.llm = _LLM()

        workbench = _Workbench()
        _unload_embeddings(workbench)
        _unload_llm(workbench)
        self.assertEqual(
            workbench.retriever.embeddings.calls,
            1,
        )
        self.assertEqual(workbench.llm.calls, 1)

    def test_preflight_checks_and_snapshots_sqlite_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime"
            database = runtime / "knowledge.sqlite3"
            db = KnowledgeDB(database)
            try:
                doc_id = db.upsert_document(
                    root / "book.pdf",
                    "abc",
                    "book",
                    1,
                )
                db.replace_page_chunks(
                    doc_id,
                    1,
                    ["肺结节测试证据"],
                )
                fake = SimpleNamespace(
                    db=db,
                    paths=SimpleNamespace(
                        database=database,
                        runtime_root=runtime,
                    ),
                )
                runtime_preflight._database_quick_check(fake)
                snapshot = runtime_preflight._database_snapshot(fake)
                self.assertIsNotNone(snapshot)
                self.assertTrue(Path(snapshot).is_file())
                self.assertNotEqual(
                    Path(snapshot).resolve(),
                    database.resolve(),
                )
                self.assertTrue(
                    db.search_lexical("肺结节", limit=5)
                )
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
