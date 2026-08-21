from __future__ import annotations

import inspect
import unittest

from phoenix_knowledge import translation_storage_hardening as storage


class TranslationStorageDefaultsTests(unittest.TestCase):
    def test_compact_builder_defaults_to_no_split(self):
        signature = inspect.signature(storage._compact_translation_pdf_build)
        self.assertEqual(signature.parameters["part_pages"].default, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
