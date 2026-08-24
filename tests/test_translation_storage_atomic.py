from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge import translation_storage_hardening as storage


class _RecordingDoc:
    def __init__(self):
        self.calls: list[dict] = []

    def save(self, path: str, **kwargs):
        self.calls.append(dict(kwargs))
        Path(path).write_bytes(b"%PDF-1.4\n%%EOF\n")


class TranslationStorageAtomicTests(unittest.TestCase):
    def test_compact_save_is_atomic_and_avoids_heavy_garbage_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "result.pdf"
            doc = _RecordingDoc()

            storage._atomic_pdf_save(doc, target)

            self.assertTrue(target.is_file())
            self.assertTrue(doc.calls)
            self.assertEqual(int(doc.calls[0].get("garbage", -1)), 0)
            self.assertNotEqual(int(doc.calls[0].get("garbage", -1)), 3)
            self.assertFalse(target.with_name("result.tmp.pdf").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
