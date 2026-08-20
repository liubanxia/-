from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.legacy_ppt import LegacyPPTConverter
from phoenix_knowledge.product_document_ingest import ProductDocumentIngestor, SUPPORTED_EXTENSIONS


def _paths(root: Path) -> WorkbenchPaths:
    return WorkbenchPaths(
        project_root=root,
        source_root=root / "library",
        runtime_root=root / "runtime",
        evidence_root=root / "evidence",
        model_root=root / "models",
        database=root / "runtime" / "knowledge.sqlite3",
        structure_root=root / "runtime" / "structure",
    ).ensure()


def _write_pptx(path: Path) -> None:
    slide = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>老式课件肺癌毛刺征</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>
</Relationships>"""
    png = b"\x89PNG\r\n\x1a\nlegacy-image"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ppt/slides/slide1.xml", slide)
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", rels)
        zf.writestr("ppt/media/image1.png", png)


class LegacyPPTProductTests(unittest.TestCase):
    def test_ppt_is_a_first_class_supported_input(self):
        self.assertIn(".ppt", SUPPORTED_EXTENSIONS)

    def test_bundled_libreoffice_is_detected_before_system_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            bundled = root / "02_开发环境" / "LibreOffice" / "program" / "soffice.exe"
            bundled.parent.mkdir(parents=True, exist_ok=True)
            bundled.write_bytes(b"stub")
            status = LegacyPPTConverter(paths).status()
            self.assertTrue(status.available)
            self.assertEqual(status.backend, "libreoffice")
            self.assertTrue(status.bundled)
            self.assertEqual(Path(status.executable), bundled.resolve())

    def test_legacy_ppt_is_indexed_after_automatic_compatibility_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            legacy = root / "teaching_2003.ppt"
            legacy.write_bytes(b"legacy-powerpoint-binary-placeholder")
            converted = root / "converted.pptx"
            _write_pptx(converted)

            db = KnowledgeDB(paths.database)
            try:
                ingestor = ProductDocumentIngestor(db, paths)
                with patch.object(ingestor.legacy_ppt, "convert", return_value=converted):
                    result = ingestor.ingest(
                        legacy,
                        copy_into_library=False,
                    )
                self.assertEqual(result.pages_total, 1)
                self.assertEqual(result.image_count, 1)
                self.assertEqual(result.copied_to_library, legacy.resolve())
                rows = db.search_lexical("肺癌毛刺征", limit=10)
                self.assertTrue(rows)
                joined = "\n".join(str(row["text"]) for row in rows)
                self.assertIn("PPT 幻灯片 1", joined)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
