from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.translation_models import (
    QualityReport,
    TranslationAttempt,
    TranslationDecision,
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


def _source_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=420, height=600)
    page.insert_text((40, 60), "CT demonstrates a 12 mm lesion in the right kidney.")
    doc.save(path)
    doc.close()


def _decision(_text: str) -> TranslationDecision:
    translated = "CT显示右肾12 mm病灶。"
    quality = QualityReport(True, 1.0, ())
    attempt = TranslationAttempt(
        backend="storage_truth_test",
        text=translated,
        quality=quality,
    )
    return TranslationDecision(
        text=translated,
        backend="storage_truth_test",
        quality=quality,
        needs_review=False,
        attempts=(attempt,),
    )


class TranslationStorageTruthTests(unittest.TestCase):
    def test_no_split_checkpoint_and_progress_are_truthful(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "book.pdf"
            _source_pdf(source)
            translator = PDFTranslator(_paths(root), _UnavailableLLM())
            translator.engine.active_backends = (
                lambda target_language="中文", smart_level="smart1": [object()]
            )
            translator.engine.available_backends = lambda: ["storage_truth_test"]
            translator.engine.unload = lambda: None
            translator.engine.translate = (
                lambda text, target_language="中文", smart_level="smart1": _decision(text)
            )
            messages: list[str] = []

            result = translator.translate_book(
                source,
                target_language="中文",
                export_format=EXPORT_PDF,
                part_pages=0,
                progress=lambda _done, _total, message: messages.append(str(message)),
            )

            self.assertEqual(result.part_pages, 0)
            self.assertFalse(any("PDF分册已完成" in item for item in messages))
            # No-split applies to more than one PDF layout, so the stable public
            # wording is deliberately layout-neutral: one complete PDF is built
            # and validated, rather than falsely promising every layout is the
            # compact in-place layout.
            self.assertTrue(any("完整PDF" in item for item in messages))

            checkpoint = next(
                translator.output_root.rglob("checkpoint.json")
            )
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(int(payload.get("part_pages", -1)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
