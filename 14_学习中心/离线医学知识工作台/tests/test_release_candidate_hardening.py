from __future__ import annotations

import math
import os
import struct
import tempfile
import time
import unittest
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from phoenix_knowledge.answerer import _locator as answer_locator
from phoenix_knowledge.compute_gateway import ComputeGateway
from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.ingest import LibraryIngestor
from phoenix_knowledge.legacy_ppt import LegacyPPTConverter
from phoenix_knowledge.pdf_parser import iter_pdf_pages_with_ocr
from phoenix_knowledge.retrieval import EmbeddingEngine, Evidence
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


class _SmartLLM:
    def available(self, profile=None):
        return True

    def generate(self, prompt, max_new_tokens=1200, *, profile=None):
        return "肺结节。"


class _GoodFallback:
    name = "fallback_good"

    def available(self):
        return True

    def translate(self, text):
        return "CT显示12 mm肺结节，衰减值45 HU，并见胸膜牵拉。"

    def unload(self):
        return None


class _Unavailable:
    name = "unavailable"

    def available(self):
        return False

    def unload(self):
        return None


class _VectorDB:
    def __init__(self, rows=20000, dim=64):
        self.calls = 0
        self.data = []
        for index in range(rows):
            vector = np.zeros(dim, dtype=np.float32)
            vector[index % dim] = 1.0
            self.data.append({"chunk_id": index + 1, "dim": dim, "vector": vector.tobytes()})

    def iter_embeddings(self, model_name):
        self.calls += 1
        return iter(self.data)


class ReleaseCandidateHardeningTests(unittest.TestCase):
    def test_compute_gateway_imports_and_flag_is_valid(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PHOENIX_KNOWLEDGE_ALLOW_REMOTE": "1"},
            clear=False,
        ):
            gateway = ComputeGateway(_paths(Path(temp)))
            self.assertTrue(gateway.remote_allowed())

    def test_failed_smart_translation_really_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = MultiModelTranslationEngine(_paths(Path(temp)), _SmartLLM())
            engine.marian = _GoodFallback()
            engine.nllb = _Unavailable()
            result = engine.translate(
                "CT showed a 12 mm pulmonary nodule measuring 45 HU with pleural retraction.",
                "中文",
            )
            self.assertEqual(result.backend, "fallback_good")
            self.assertTrue(result.quality.ok)
            self.assertEqual(len(result.attempts), 2)
            self.assertFalse(result.attempts[0].quality.ok)

    def test_powershell_alone_never_claims_powerpoint_is_available(self):
        with tempfile.TemporaryDirectory() as temp:
            converter = LegacyPPTConverter(_paths(Path(temp)))
            with patch.object(converter, "_find_libreoffice", return_value=None), patch.object(
                converter, "_powerpoint_registered", return_value=False
            ), patch.object(converter, "_find_powershell", return_value=Path("powershell.exe")):
                status = converter.status()
            self.assertFalse(status.available)
            self.assertEqual(status.backend, "unavailable")

    def test_mixed_document_locator_uses_real_unit_names(self):
        ppt = Evidence(1, "D1", "课件", "lecture.ppt", 7, "x", 1.0)
        pptx = Evidence(2, "D2", "课件2", "lecture.pptx", 8, "x", 1.0)
        docx = Evidence(3, "D3", "文档", "note.docx", 3, "x", 1.0)
        self.assertEqual(answer_locator(ppt), "第7张幻灯片")
        self.assertEqual(answer_locator(pptx), "第8张幻灯片")
        self.assertEqual(answer_locator(docx), "文档单元3")

    def test_scan_pdf_ocr_text_is_written_back_when_local_ocr_works(self):
        with tempfile.TemporaryDirectory() as temp:
            pdf = Path(temp) / "scan.pdf"
            _scan_pdf(pdf)
            with patch("phoenix_knowledge.pdf_parser._ocr_page_text", return_value="OCR识别：肺结节12 mm"):
                pages = list(iter_pdf_pages_with_ocr(pdf))
            self.assertEqual(len(pages), 1)
            self.assertTrue(pages[0].ocr_attempted)
            self.assertTrue(pages[0].ocr_used)
            self.assertIn("肺结节", pages[0].text)

    def test_scan_pdf_without_ocr_is_explicitly_ocr_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "scan.pdf"
            _scan_pdf(pdf)
            paths = _paths(root)
            db = KnowledgeDB(paths.database)
            try:
                ingestor = LibraryIngestor(db, paths)
                with patch("phoenix_knowledge.pdf_parser._ocr_page_text", side_effect=RuntimeError("missing tessdata")):
                    result = ingestor.ingest_pdf(pdf, copy_into_library=False, extract_images=False)
                row = db.get_document(result.document_id)
                self.assertEqual(str(row["status"]), "ocr_required")
                self.assertIn("OCR_REQUIRED", result.warning)
            finally:
                db.close()

    def test_pdf_and_docx_exports_embed_real_images(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "topic_assets"
            assets.mkdir()
            (assets / "figure.png").write_bytes(_png_bytes())
            source = root / "topic.md"
            source.write_text(
                "# 肺结节\n\n[S1] 结节边缘可评估。\n\n![原图](topic_assets/figure.png)\n",
                encoding="utf-8",
            )
            bundle = MultiFormatExporter(root / "out").export_path(source, title="测试专题")

            with zipfile.ZipFile(bundle.docx) as zf:
                self.assertTrue(any(name.startswith("word/media/") for name in zf.namelist()))
            pdf = fitz.open(bundle.pdf)
            try:
                self.assertTrue(any(page.get_images(full=True) for page in pdf))
            finally:
                pdf.close()

    def test_vector_search_has_real_reuse_and_latency_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            db = _VectorDB(rows=20000, dim=64)
            engine = EmbeddingEngine(db, _paths(Path(temp)))
            query = np.ones(64, dtype=np.float32) / math.sqrt(64)
            engine._encode_query = lambda _query: query
            start = time.perf_counter()
            first = engine.search("肺结节", limit=20)
            first_elapsed = time.perf_counter() - start
            start = time.perf_counter()
            second = engine.search("肺结节", limit=20)
            second_elapsed = time.perf_counter() - start
            self.assertTrue(first and second)
            self.assertEqual(db.calls, 1)
            self.assertLess(first_elapsed, 4.0)
            self.assertLess(second_elapsed, 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
