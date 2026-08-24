from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.workbench import MedicalKnowledgeWorkbench


class PdfIngestSmokeTest(unittest.TestCase):
    def test_real_pdf_ingest_search_and_evidence_only_answer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "Chest_CT_Test.pdf"

            document = fitz.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "Lung nodule spiculation and pleural retraction are imaging findings.",
            )
            document.save(str(pdf_path))
            document.close()

            runtime = root / "runtime"
            paths = WorkbenchPaths(
                project_root=root,
                source_root=root / "pdf_library",
                runtime_root=runtime,
                evidence_root=root / "evidence",
                model_root=root / "models",
                database=runtime / "knowledge.sqlite3",
                structure_root=runtime / "docling",
            ).ensure()

            workbench = MedicalKnowledgeWorkbench(paths)
            try:
                result = workbench.ingest(pdf_path)
                self.assertEqual(result.pages_total, 1)
                self.assertEqual(result.pages_indexed, 1)

                answer = workbench.ask(
                    "lung nodule spiculation"
                )
                self.assertEqual(answer.mode, "evidence_only")
                self.assertTrue(answer.evidence)
                self.assertIn("Chest_CT_Test", answer.text)
                self.assertIn("第1页", answer.text)
                self.assertIn("spiculation", answer.text)
            finally:
                workbench.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
