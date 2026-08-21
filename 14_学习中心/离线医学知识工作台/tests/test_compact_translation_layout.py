from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from phoenix_knowledge.translation_layout_compact import (
    LAYOUT_SOURCE_TRANSLATED,
)
from phoenix_knowledge.translation_pdf import TranslationPDFBuilder


class CompactTranslationLayoutTests(unittest.TestCase):
    @staticmethod
    def _jpeg(seed: int) -> bytes:
        rng = np.random.default_rng(seed)
        pixels = rng.integers(
            0,
            256,
            size=(1000, 1400, 3),
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=94, optimize=False)
        return buffer.getvalue()

    @classmethod
    def _make_source(cls, path: Path, pages: int = 4) -> None:
        import fitz

        doc = fitz.open()
        try:
            for index in range(pages):
                page = doc.new_page(width=595, height=842)
                page.insert_textbox(
                    fitz.Rect(44, 44, 551, 150),
                    (
                        f"Chapter {index + 1}. CT demonstrates no pleural effusion. "
                        "A 12 mm lesion is present in the right kidney. "
                        "Pulmonary nodule assessment should preserve measurements."
                    ),
                    fontsize=11,
                )
                page.insert_image(
                    fitz.Rect(44, 175, 551, 760),
                    stream=cls._jpeg(20260821 + index),
                )
                page.insert_text((285, 815), str(index + 1), fontsize=8)
            doc.save(
                path,
                garbage=2,
                deflate=True,
                deflate_images=True,
                use_objstms=1,
            )
        finally:
            doc.close()

    @staticmethod
    def _write_translations(root: Path, pages: int) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for page in range(1, pages + 1):
            (root / f"{page:06d}.txt").write_text(
                (
                    f"第{page}章。CT显示无胸腔积液。右肾可见12 mm病变。"
                    "肺结节评估应保留原始测量值。"
                ),
                encoding="utf-8",
            )

    def test_original_images_are_kept_and_english_text_is_replaced_in_place(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            output_root = root / "out"
            self._make_source(source, pages=4)
            self._write_translations(pages_root, pages=4)

            complete, parts = TranslationPDFBuilder(
                source,
                pages_root,
                output_root,
            ).build(
                start_page=1,
                total_pages=4,
                layout=LAYOUT_SOURCE_TRANSLATED,
                part_pages=0,
            )

            self.assertTrue(complete.is_file())
            self.assertEqual(parts, ())

            src = fitz.open(source)
            out = fitz.open(complete)
            try:
                self.assertEqual(out.page_count, src.page_count)
                for index in range(src.page_count):
                    self.assertAlmostEqual(
                        float(out[index].rect.width),
                        float(src[index].rect.width),
                        places=2,
                    )
                    self.assertAlmostEqual(
                        float(out[index].rect.height),
                        float(src[index].rect.height),
                        places=2,
                    )
                    self.assertEqual(
                        len(out[index].get_images(full=True)),
                        len(src[index].get_images(full=True)),
                    )

                text = "\n".join(page.get_text("text") for page in out)
                self.assertIn("右肾", text)
                self.assertIn("12 mm", text)
                self.assertNotIn("Pulmonary nodule assessment", text)
            finally:
                out.close()
                src.close()

            source_size = int(source.stat().st_size)
            output_size = int(complete.stat().st_size)
            budget = max(
                int(source_size * 1.18),
                source_size + int(2.5 * 1024 * 1024),
            )
            self.assertLessEqual(output_size, budget)

            report = json.loads(
                (output_root / "PDF体积报告.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["mode"], LAYOUT_SOURCE_TRANSLATED)
            self.assertEqual(report["split_volumes"], 0)
            self.assertEqual(
                report["pages_with_footer_overflow_or_scan_fallback"],
                0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
