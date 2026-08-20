from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.translation_pdf import (
    LAYOUT_TRANSLATED_ONLY,
    TranslationPDFBuilder,
)


class ReleaseTranslationLayoutTests(unittest.TestCase):
    def test_long_chinese_translation_remains_complete_in_pdf(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages = root / "pages"
            output = root / "out"
            pages.mkdir()

            doc = fitz.open()
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(
                fitz.Rect(48, 60, 545, 760),
                "Source medical page",
                fontsize=12,
            )
            doc.save(source)
            doc.close()

            long_text = (
                "肺结节影像学征象包括边缘、密度、内部结构及邻近组织改变。"
                "所有数字、单位、否定和侧别都必须完整保留。"
            ) * 70
            marker = "最终完整性标记12345"
            (pages / "000001.txt").write_text(
                long_text + marker,
                encoding="utf-8",
            )

            complete, parts = TranslationPDFBuilder(
                source,
                pages,
                output,
            ).build(
                start_page=1,
                total_pages=1,
                layout=LAYOUT_TRANSLATED_ONLY,
                part_pages=50,
            )

            self.assertTrue(complete.is_file())
            self.assertEqual(len(parts), 1)
            rendered = fitz.open(complete)
            try:
                self.assertEqual(rendered.page_count, 1)
                self.assertGreater(
                    float(rendered[0].rect.height),
                    842.0,
                )
                extracted = rendered[0].get_text("text")
            finally:
                rendered.close()
            self.assertIn(marker, extracted)
            self.assertIn("肺结节影像学征象", extracted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
