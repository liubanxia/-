from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.ingest import LibraryIngestor


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


def _png_bytes() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
    raw = b"".join([b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4)])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _scan_pdf(path: Path) -> None:
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_image(fitz.Rect(20, 20, 280, 280), stream=_png_bytes())
    doc.save(path)
    doc.close()


class OCRRetryTests(unittest.TestCase):
    def test_ocr_required_document_reprocesses_after_ocr_becomes_available(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "scan.pdf"
            _scan_pdf(pdf)
            paths = _paths(root)
            db = KnowledgeDB(paths.database)
            ingestor = LibraryIngestor(db, paths)
            try:
                with patch("phoenix_knowledge.pdf_parser._ocr_page_text", side_effect=RuntimeError("missing tessdata")):
                    first = ingestor.ingest_pdf(pdf, copy_into_library=False, extract_images=False)
                first_row = db.get_document(first.document_id)
                self.assertEqual(str(first_row["status"]), "ocr_required")

                with patch("phoenix_knowledge.pdf_parser._ocr_page_text", return_value="OCR识别成功：右肺结节12 mm"):
                    second = ingestor.ingest_pdf(pdf, copy_into_library=False, extract_images=False)
                second_row = db.get_document(second.document_id)
                self.assertEqual(str(second_row["status"]), "indexed")
                hits = db.search_lexical("肺结节", limit=5)
                self.assertTrue(hits)
                self.assertNotIn("OCR_REQUIRED", second.warning)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
