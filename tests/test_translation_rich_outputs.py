from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.translator import PDFTranslator


class _LLM:
    def available(self):
        return False


def _paths(tmp_path: Path) -> WorkbenchPaths:
    return WorkbenchPaths(
        project_root=tmp_path,
        source_root=tmp_path / "sources",
        runtime_root=tmp_path / "runtime",
        evidence_root=tmp_path / "evidence",
        model_root=tmp_path / "models",
        database=tmp_path / "runtime" / "knowledge.sqlite3",
        structure_root=tmp_path / "runtime" / "structure",
    ).ensure()


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


class TranslationRichOutputTests(unittest.TestCase):
    def test_translation_assembles_txt_markdown_html_and_images(self):
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

            translator = PDFTranslator(_paths(tmp_path), _LLM())
            translator.assets.extract(pdf)

            book_root = tmp_path / "out"
            pages_root = book_root / "pages"
            pages_root.mkdir(parents=True)
            (pages_root / "000001.txt").write_text("这是第一页译文。", encoding="utf-8")

            outputs, image_count = translator._assemble_outputs(
                pdf,
                book_root,
                pages_root,
                1,
                1,
                "中文",
            )

            suffixes = {path.suffix for path in outputs}
            self.assertTrue({".txt", ".md", ".html"}.issubset(suffixes))
            self.assertGreaterEqual(image_count, 1)
            self.assertTrue((book_root / "images").is_dir())

            markdown = next(path for path in outputs if path.suffix == ".md").read_text(encoding="utf-8")
            html = next(path for path in outputs if path.suffix == ".html").read_text(encoding="utf-8")
            self.assertIn("这是第一页译文", markdown)
            self.assertIn("![第1页原图", markdown)
            self.assertIn("<img", html)


if __name__ == "__main__":
    unittest.main()
