from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.notes import TXTNotesOrganizer


class DummyLLM:
    def available(self):
        return False


class TXTNotesTest(unittest.TestCase):
    def test_without_llm_preserves_source_and_saves_txt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = WorkbenchPaths(
                project_root=root,
                source_root=root / "pdf",
                runtime_root=root / "runtime",
                evidence_root=root / "evidence",
                model_root=root / "models",
                database=root / "runtime" / "knowledge.sqlite3",
                structure_root=root / "runtime" / "docling",
            ).ensure()
            organizer = TXTNotesOrganizer(paths, DummyLLM())
            source = "肺结节：12 mm。\nCT值约35 HU。\n需要鉴别诊断。"
            result = organizer.organize(
                source,
                title="胸部CT笔记",
            )
            self.assertEqual(result.mode, "source_only")
            self.assertTrue(result.output_path.is_file())
            self.assertIn("12 mm", result.text)
            self.assertIn("35 HU", result.text)
            self.assertIn("肺结节", result.text)

    def test_organize_file_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = WorkbenchPaths(
                project_root=root,
                source_root=root / "pdf",
                runtime_root=root / "runtime",
                evidence_root=root / "evidence",
                model_root=root / "models",
                database=root / "runtime" / "knowledge.sqlite3",
                structure_root=root / "runtime" / "docling",
            ).ensure()
            source_file = root / "note.txt"
            source_file.write_text("\ufeffMRI T2WI高信号。", encoding="utf-8")
            organizer = TXTNotesOrganizer(paths, DummyLLM())
            result = organizer.organize_file(source_file)
            self.assertIn("MRI", result.text)
            self.assertIn("T2WI", result.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
