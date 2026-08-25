from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.translation_models import MultiModelTranslationEngine
from test_release_candidate_hardening import (
    _GoodFallback,
    _SmartLLM,
    _Unavailable,
    _paths,
)


class V3CandidateBlockingTest(unittest.TestCase):
    def test_failed_smart2_never_promotes_preview_fallback_or_unverified_draft(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = MultiModelTranslationEngine(_paths(Path(temp)), _SmartLLM())
            engine.marian = _GoodFallback()
            engine.nllb = _Unavailable()

            result = engine.translate(
                "CT showed a 12 mm pulmonary nodule measuring 45 HU with pleural retraction.",
                "中文",
                smart_level="smart2",
            )

            self.assertNotEqual(result.backend, "fallback_good")
            self.assertTrue(str(result.backend).startswith("blocked_local_candidate:"))
            self.assertFalse(result.quality.ok)
            self.assertTrue(result.needs_review)
            self.assertGreaterEqual(len(result.attempts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
