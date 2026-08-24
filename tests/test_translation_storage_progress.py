from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.translation_pdf import (
    LAYOUT_TRANSLATED_ONLY,
    TranslationPDFBuilder,
)


class TranslationStorageProgressTests(unittest.TestCase):
    def test_size_report_mentions_no_duplicate_split_storage(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            pages_root.mkdir()
            doc = fitz.open()
            page = doc.new_page(width=420, height=600)
            page.insert_text((40, 60), "CT demonstrates a 12 mm lesion.")
            doc.save(source)
            doc.close()
            (pages_root / "000001.txt").write_text(
                "CT显示12 mm病灶。",
                encoding="utf-8",
            )

            messages = []
            TranslationPDFBuilder(
                source,
                pages_root,
                root / "out",
            ).build(
                start_page=1,
                total_pages=1,
                layout=LAYOUT_TRANSLATED_ONLY,
                part_pages=0,
                progress=lambda _d, _t, message: messages.append(str(message)),
            )

            text = "\n".join(messages)
            self.assertIn("体积比", text)
            self.assertIn("未生成重复分册", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
