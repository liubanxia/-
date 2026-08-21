from __future__ import annotations

import inspect
import unittest

from phoenix_knowledge import translation_storage_hardening as storage


class TranslationStorageNoRasterizeTests(unittest.TestCase):
    def test_compact_path_uses_insert_pdf_for_normal_pages(self):
        source = inspect.getsource(storage._append_original_page_compact)
        self.assertIn("insert_pdf", source)
        self.assertNotIn("get_pixmap", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
