from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.document_ingest import MultiDocumentIngestor
from phoenix_knowledge.rich_export import MultiFormatExporter


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
    slide1 = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <p:cSld><p:spTree>
  <p:sp><p:txBody><a:p><a:r><a:t>肺结节毛刺征</a:t></a:r></a:p></p:txBody></p:sp>
 </p:spTree></p:cSld>
</p:sld>"""
    slide2 = slide1.replace("肺结节毛刺征", "胸膜凹陷征")
    rels1 = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>
</Relationships>"""
    notes = """<?xml version="1.0" encoding="UTF-8"?>
<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>注意与炎症鉴别</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:notes>"""
    png = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x0f\x9b\x8f"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ppt/slides/slide1.xml", slide1)
        zf.writestr("ppt/slides/slide2.xml", slide2)
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", rels1)
        zf.writestr("ppt/notesSlides/notesSlide1.xml", notes)
        zf.writestr("ppt/media/image1.png", png)


def _write_docx(path: Path) -> None:
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <w:body>
  <w:p><w:r><w:t>肝细胞癌动脉期强化</w:t></w:r></w:p>
  <w:p><w:r><w:t>门静脉期洗脱</w:t></w:r></w:p>
 </w:body>
</w:document>"""
    types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", types)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/_rels/document.xml.rels", rels)


class MultiDocumentIngestTests(unittest.TestCase):
    def test_pptx_keeps_slides_notes_and_page_image_relation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            source = root / "lecture.pptx"
            _write_pptx(source)
            db = KnowledgeDB(paths.database)
            try:
                result = MultiDocumentIngestor(db, paths).ingest(
                    source,
                    copy_into_library=False,
                )
                self.assertEqual(result.pages_total, 2)
                self.assertEqual(result.image_count, 1)
                rows = db.search_lexical("肺结节", limit=10)
                self.assertTrue(rows)
                text = "\n".join(str(row["text"]) for row in rows)
                self.assertIn("幻灯片 1", text)
                self.assertIn("注意与炎症鉴别", text)
            finally:
                db.close()

    def test_docx_txt_and_markdown_are_indexable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            db = KnowledgeDB(paths.database)
            try:
                docx = root / "liver.docx"
                _write_docx(docx)
                txt = root / "note.txt"
                txt.write_text("肾结石 CT 高密度灶", encoding="utf-8")
                md = root / "note.md"
                md.write_text("# 肠梗阻\n移行区与近端肠管扩张", encoding="utf-8")
                ingestor = MultiDocumentIngestor(db, paths)
                ingestor.ingest(docx, copy_into_library=False)
                ingestor.ingest(txt, copy_into_library=False)
                ingestor.ingest(md, copy_into_library=False)
                self.assertTrue(db.search_lexical("肝细胞癌", limit=10))
                self.assertTrue(db.search_lexical("肾结石", limit=10))
                self.assertTrue(db.search_lexical("肠梗阻", limit=10))
            finally:
                db.close()


class MultiFormatExportTests(unittest.TestCase):
    def test_export_generates_pdf_docx_markdown_and_txt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exporter = MultiFormatExporter(root / "out")
            bundle = exporter.export_text(
                "# 肺结节专题\n\n- 毛刺征 [S12]\n- 胸膜凹陷征 [S15]\n",
                title="肺结节专题",
            )
            for path in bundle.output_paths:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0)
            with zipfile.ZipFile(bundle.docx) as zf:
                self.assertIn("word/document.xml", zf.namelist())
            self.assertIn("[S12]", bundle.text.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
