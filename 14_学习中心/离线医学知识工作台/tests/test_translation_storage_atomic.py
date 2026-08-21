from __future__ import annotations

import inspect
import unittest

from phoenix_knowledge import translation_storage_hardening as storage


class TranslationStorageAtomicTests(unittest.TestCase):
    def test_compact_save_is_atomic_and_avoids_heavy_garbage_pass(self):
        source = inspect.getsource(storage._atomic_pdf_save)
        self.assertIn("os.replace", source)
        self.assertIn('"garbage": 0', source)
        self.assertNotIn("garbage=3", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
