from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.translator import PDFTranslator


class TranslationCheckpointResumeTest(unittest.TestCase):
    def test_existing_checkpoint_start_page_is_authoritative(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "source_sha256": "abc",
                        "target_language": "中文",
                        "start_page": 37,
                        "last_completed_page": 42,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            state = PDFTranslator._read_json(checkpoint)
            resolved = PDFTranslator._resolve_resume_start_page(
                state,
                1,
                force_restart=False,
            )
            self.assertEqual(resolved, 37)

    def test_force_restart_keeps_requested_start_page(self):
        state = {"start_page": 37}
        self.assertEqual(
            PDFTranslator._resolve_resume_start_page(
                state,
                12,
                force_restart=True,
            ),
            12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
