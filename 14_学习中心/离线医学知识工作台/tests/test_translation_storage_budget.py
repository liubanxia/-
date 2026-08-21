from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.translation_models import (
    QualityReport,
    TranslationAttempt,
    TranslationDecision,
)
from phoenix_knowledge.translation_pdf import (
    LAYOUT_ORIGINAL_BILINGUAL,
    TranslationPDFBuilder,
)
from phoenix_knowledge.translator import EXPORT_PDF, PDFTranslator


class _UnavailableLLM:
    def available(self, *args, **kwargs):
        return False

    def unload(self):
        return None


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


class TranslationStorageBudgetTests(unittest.TestCase):
    @staticmethod
    def _noise_png() -> bytes:
        rng = np.random.default_rng(20260821)
        pixels = rng.integers(
            0,
            256,
            size=(1200, 1200, 3),
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=6)
        return buffer.getvalue()

    @classmethod
    def _make_source(cls, path: Path, pages: int = 3) -> None:
        import fitz

        image = cls._noise_png()
        doc = fitz.open()
        try:
            xref = 0
            for index in range(pages):
                page = doc.new_page(width=595, height=842)
                page.insert_text(
                    (40, 45),
                    f"Source page {index + 1}: CT demonstrates a 12 mm lesion.",
                    fontsize=11,
                )
                rect = fitz.Rect(40, 80, 555, 760)
                if xref:
                    page.insert_image(rect, xref=xref)
                else:
                    xref = page.insert_image(rect, stream=image)
            doc.save(path, deflate=True)
        finally:
            doc.close()

    @staticmethod
    def _write_pages(root: Path, pages: int) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for page in range(1, pages + 1):
            (root / f"{page:06d}.txt").write_text(
                f"第{page}页中文译文：CT显示右肾12 mm病灶，无胸腔积液。",
                encoding="utf-8",
            )

    @staticmethod
    def _fake_translation(text: str) -> TranslationDecision:
        translated = "CT显示右肾12 mm病灶，无胸腔积液。"
        quality = QualityReport(True, 1.0, ())
        attempt = TranslationAttempt(
            backend="storage_test",
            text=translated,
            quality=quality,
        )
        return TranslationDecision(
            text=translated,
            backend="storage_test",
            quality=quality,
            needs_review=False,
            attempts=(attempt,),
        )

    def test_default_compact_bilingual_output_reuses_source_and_skips_parts(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            outputs = root / "outputs"
            self._make_source(source, pages=3)
            self._write_pages(pages_root, pages=3)

            complete, parts = TranslationPDFBuilder(
                source,
                pages_root,
                outputs,
            ).build(
                start_page=1,
                total_pages=3,
                layout=LAYOUT_ORIGINAL_BILINGUAL,
                part_pages=0,
            )

            self.assertTrue(complete.is_file())
            self.assertEqual(parts, ())
            self.assertFalse((outputs / "PDF分册").exists())

            source_size = source.stat().st_size
            output_size = complete.stat().st_size
            self.assertLess(
                output_size,
                source_size + 3 * 1024 * 1024,
            )

            source_doc = fitz.open(source)
            translated_doc = fitz.open(complete)
            try:
                self.assertEqual(
                    translated_doc.page_count,
                    source_doc.page_count,
                )
                self.assertGreater(
                    translated_doc[0].rect.height,
                    source_doc[0].rect.height,
                )
                text = translated_doc[0].get_text("text")
                self.assertIn("Source page 1", text)
                self.assertIn("12", text)
            finally:
                translated_doc.close()
                source_doc.close()

    def test_split_volumes_exist_only_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            outputs = root / "outputs"
            self._make_source(source, pages=2)
            self._write_pages(pages_root, pages=2)

            complete, parts = TranslationPDFBuilder(
                source,
                pages_root,
                outputs,
            ).build(
                start_page=1,
                total_pages=2,
                layout=LAYOUT_ORIGINAL_BILINGUAL,
                part_pages=1,
            )
            self.assertTrue(complete.is_file())
            self.assertEqual(len(parts), 2)
            self.assertTrue(all(path.is_file() for path in parts))

            complete2, parts2 = TranslationPDFBuilder(
                source,
                pages_root,
                outputs,
            ).build(
                start_page=1,
                total_pages=2,
                layout=LAYOUT_ORIGINAL_BILINGUAL,
                part_pages=0,
            )
            self.assertTrue(complete2.is_file())
            self.assertEqual(parts2, ())
            self.assertFalse((outputs / "PDF分册").exists())

    def test_real_translator_zero_does_not_secretly_reenable_split_volumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "book.pdf"
            self._make_source(source, pages=2)
            translator = PDFTranslator(_paths(root), _UnavailableLLM())
            translator.engine.active_backends = (
                lambda target_language="中文", smart_level="smart1": [object()]
            )
            translator.engine.available_backends = lambda: ["storage_test"]
            translator.engine.unload = lambda: None
            translator.engine.translate = (
                lambda text, target_language="中文", smart_level="smart1": self._fake_translation(text)
            )

            result = translator.translate_book(
                source,
                target_language="中文",
                export_format=EXPORT_PDF,
                part_pages=0,
            )

            self.assertFalse(result.paused)
            self.assertEqual(result.part_pages, 0)
            pdfs = [
                Path(path)
                for path in result.output_paths
                if Path(path).suffix.lower() == ".pdf"
            ]
            self.assertEqual(len(pdfs), 1)
            self.assertTrue(pdfs[0].is_file())
            self.assertFalse(pdfs[0].parent.joinpath("PDF分册").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
