from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.answerer import KnowledgeAnswerer
from phoenix_knowledge.pdf_assets import PDFAssetStore
from phoenix_knowledge.retrieval import Evidence


def _png_bytes() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_scanline = b"\x00\xff\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw_scanline))
        + chunk(b"IEND", b"")
    )


class _Retriever:
    def search(self, query, limit=18, use_embeddings=True):
        return [
            Evidence(
                chunk_id=7,
                source_key="D1",
                title="Test Book",
                path="test.pdf",
                page=12,
                text="Pulmonary nodule margins and attenuation should be assessed.",
                score=1.0,
            )
        ]


class _SlowLLM:
    def __init__(self):
        self.called = False

    def available(self):
        return True

    def generate(self, prompt, max_new_tokens=1600):
        self.called = True
        return "结节边缘与密度需要评估。[S7]"


class FastQAAndRichMediaTests(unittest.TestCase):
    def test_pdf_qa_defaults_to_fast_evidence_mode(self):
        with patch.dict(os.environ, {"PHOENIX_KNOWLEDGE_DEEP_QA": "0"}, clear=False):
            llm = _SlowLLM()
            answerer = KnowledgeAnswerer(_Retriever(), llm)
            result = answerer.ask("肺结节看什么")
            self.assertEqual(result.mode, "evidence_only")
            self.assertIn("[S7]", result.text)
            self.assertFalse(llm.called)

    def test_pdf_qa_can_enable_deep_generation(self):
        llm = _SlowLLM()
        answerer = KnowledgeAnswerer(_Retriever(), llm)
        result = answerer.ask("肺结节看什么", deep=True)
        self.assertEqual(result.mode, "grounded_generation")
        self.assertTrue(llm.called)
        self.assertIn("[S7]", result.text)

    def test_pdf_asset_store_extracts_embedded_images(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            pdf = tmp_path / "book.pdf"
            doc = fitz.open()
            page = doc.new_page(width=200, height=200)
            page.insert_text((20, 30), "Figure 1")
            page.insert_image(fitz.Rect(20, 50, 120, 150), stream=_png_bytes())
            doc.save(pdf)
            doc.close()

            store = PDFAssetStore(tmp_path / "runtime")
            manifest = store.extract(pdf)
            self.assertGreaterEqual(manifest["image_count"], 1)
            assets = store.page_assets(pdf, 1)
            self.assertTrue(assets)
            self.assertTrue(assets[0].path.is_file())


if __name__ == "__main__":
    unittest.main()
