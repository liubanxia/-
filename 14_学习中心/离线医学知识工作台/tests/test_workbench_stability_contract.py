from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phoenix_knowledge.answerer import AnswerResult
from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.output_contracts import (
    OutputContractError,
    transactional_export_path,
    validate_export_bundle,
)
from phoenix_knowledge.rich_export import MultiFormatExporter
from phoenix_knowledge.workbench import MedicalKnowledgeWorkbench
from phoenix_knowledge.workbench_stability_core import architecture_status


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


class WorkbenchStabilityContractTests(unittest.TestCase):
    def test_public_workbench_topology_is_one_contract_layer(self):
        with tempfile.TemporaryDirectory() as temp:
            wb = MedicalKnowledgeWorkbench(_paths(Path(temp)))
            try:
                state = architecture_status(wb)
                self.assertTrue(state["ready"], state["broken"])
                self.assertEqual(state["workbench_wrapper_depth"], 1)
                self.assertEqual(state["translation_wrapper_depth"], 1)
                self.assertEqual(state["formal_translation_contract"], 1)
                status = wb.status()
                self.assertEqual(status["workbench_contract"], 3)
                self.assertEqual(status["office_translation_contract"], 2)
                self.assertIs(
                    wb.translator.engine,
                    wb.office_translator.engine,
                )
                self.assertTrue(
                    set(status["translation_backends"]).isdisjoint(
                        {"marian_en_zh", "nllb_600m_en_zh"}
                    )
                )
            finally:
                wb.close()

    def test_no_model_public_smoke_still_produces_real_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wb = MedicalKnowledgeWorkbench(_paths(root))
            try:
                source = root / "肺结节.txt"
                source.write_text(
                    "肺结节CT征象：边缘毛刺。病灶直径12 mm。"
                    "本段仅用于Phoenix稳定性测试。",
                    encoding="utf-8",
                )
                imported = wb.ingest(source, _defer_embeddings=True)
                self.assertTrue(Path(imported.copied_to_library).is_file())
                answer = wb.ask("肺结节12 mm", use_embeddings=False, deep=False)
                self.assertTrue(answer.text.strip())
                self.assertTrue(answer.evidence)
                output, _task_id = wb.organize(
                    "肺结节稳定性",
                    "整理肺结节CT征象和病灶大小",
                    candidate_limit=12,
                    batch_size=4,
                )
                self.assertTrue(Path(output).is_file())
                self.assertIsNotNone(wb.last_export_bundle)
                validate_export_bundle(wb.last_export_bundle)
                note = wb.organize_txt(
                    "CT提示右肺结节12 mm。",
                    title="稳定性笔记",
                )
                self.assertTrue(note.output_path.is_file())
                self.assertIn("12 mm", note.output_path.read_text(encoding="utf-8"))
                state = architecture_status(wb)
                self.assertTrue(state["ready"], state["broken"])
            finally:
                wb.close()

    def test_export_failure_cannot_report_organize_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wb = MedicalKnowledgeWorkbench(_paths(root))
            try:
                source = root / "organized.md"
                source.write_text("# result\n\n[S1] evidence\n", encoding="utf-8")
                task_id = wb.db.create_task(
                    "deep_organize",
                    {"title": "topic", "instruction": "instruction", "chunk_ids": []},
                    total=1,
                )
                wb.organizer = SimpleNamespace(
                    organize=lambda title, instruction, **kwargs: (source, task_id)
                )
                with patch(
                    "phoenix_knowledge.workbench_stability_core.transactional_export_path",
                    side_effect=RuntimeError("simulated export failure"),
                ):
                    with self.assertRaises(OutputContractError):
                        wb.organize("topic", "instruction")
                self.assertIsNone(wb.last_export_bundle)
                self.assertIn("simulated export failure", wb.last_export_error)
                task = wb.db.get_task(task_id)
                self.assertEqual(str(task["status"]), "failed")
            finally:
                wb.close()

    def test_bundle_post_publish_validation_failure_rolls_back_old_good_bundle(self):
        import phoenix_knowledge.output_contracts as contracts
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "topic.md"
            source.write_text("# Old\n\nstable content\n", encoding="utf-8")
            exporter = MultiFormatExporter(root / "out")
            first = transactional_export_path(exporter, source, title="topic")
            old_pdf = first.pdf.read_bytes()
            old_md = first.markdown.read_bytes()
            source.write_text("# New\n\nreplacement content\n", encoding="utf-8")
            real_validate = contracts.validate_export_bundle
            calls = {"n": 0}
            def fail_after_publish(bundle):
                calls["n"] += 1
                report = real_validate(bundle)
                if calls["n"] == 2:
                    raise OutputContractError("simulated post-publish reopen failure")
                return report
            with patch.object(
                contracts,
                "validate_export_bundle",
                side_effect=fail_after_publish,
            ):
                with self.assertRaises(OutputContractError):
                    contracts.transactional_export_path(exporter, source, title="topic")
            self.assertEqual(first.pdf.read_bytes(), old_pdf)
            self.assertEqual(first.markdown.read_bytes(), old_md)
            validate_export_bundle(first)

    def test_library_copy_interruption_leaves_no_partial_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wb = MedicalKnowledgeWorkbench(_paths(root))
            try:
                external = root / "external"
                external.mkdir()
                source = external / "book.txt"
                source.write_text("stable source", encoding="utf-8")
                target = wb.paths.source_root / source.name
                def broken_copy(src, dst, *args, **kwargs):
                    Path(dst).write_bytes(b"partial")
                    raise OSError("simulated copy interruption")
                with patch(
                    "phoenix_knowledge.workbench_stability_core.shutil.copy2",
                    side_effect=broken_copy,
                ):
                    with self.assertRaises(OSError):
                        wb.ingestor._library_copy(source)
                self.assertFalse(target.exists())
                self.assertFalse(any(wb.paths.source_root.glob(".pxcopy-*")))
                copied = wb.ingestor._library_copy(source)
                self.assertTrue(copied.is_file())
                self.assertEqual(copied.read_text(encoding="utf-8"), "stable source")
            finally:
                wb.close()

    def test_note_generation_failure_preserves_previous_good_note(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wb = MedicalKnowledgeWorkbench(_paths(root))
            try:
                first = wb.organize_txt("old stable note", title="稳定笔记")
                old = first.output_path.read_bytes()
                def broken(self_notes, source_text, **kwargs):
                    staged = Path(self_notes.output_root) / "稳定笔记.txt"
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_text("partial new note", encoding="utf-8")
                    raise RuntimeError("simulated note crash")
                wb.notes.organize = types.MethodType(broken, wb.notes)
                with self.assertRaises(RuntimeError):
                    wb.organize_txt("new note", title="稳定笔记")
                self.assertEqual(first.output_path.read_bytes(), old)
                self.assertFalse(any(Path(wb.notes.output_root).glob(".pxnotes-*")))
            finally:
                wb.close()

    def test_qa_semantic_failure_degrades_to_local_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            wb = MedicalKnowledgeWorkbench(_paths(Path(temp)))
            try:
                evidence = [SimpleNamespace(citation="[S1]")]
                def fake_ask(query, **kwargs):
                    if kwargs.get("use_embeddings", True):
                        raise RuntimeError("semantic runtime failed")
                    return AnswerResult(
                        text="[S1] 本地关键词证据",
                        evidence=evidence,
                        mode="evidence_only",
                    )
                wb.answerer.ask = fake_ask
                result = wb.ask("test", use_embeddings=True, deep=True)
                self.assertEqual(result.mode, "degraded_evidence_only")
                self.assertIn("降级保护", result.text)
                self.assertIn("[S1]", result.text)
            finally:
                wb.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
