from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.translation_models import (
    MultiModelTranslationEngine,
    QualityReport,
    TranslationAttempt,
    TranslationDecision,
)
from phoenix_knowledge.translation_pdf import (
    LAYOUT_ORIGINAL_BILINGUAL,
    TranslationPDFBuilder,
)
from phoenix_knowledge.translator import (
    EXPORT_TXT,
    PDFTranslator,
    _translation_chunk_chars,
)


class _SmartLLM:
    def __init__(self):
        self.profiles: list[str | None] = []
        self.max_tokens: list[int] = []

    def available(self, profile=None):
        return True

    def generate(self, prompt, max_new_tokens=1200, *, profile=None):
        self.profiles.append(profile)
        self.max_tokens.append(int(max_new_tokens))
        return "CT显示5 mm肺结节，边缘清楚。"


class _UnavailableLLM:
    def available(self, *args, **kwargs):
        return False


def _paths(tmp_path: Path) -> WorkbenchPaths:
    return WorkbenchPaths(
        project_root=tmp_path,
        source_root=tmp_path / "sources",
        runtime_root=tmp_path / "runtime",
        evidence_root=tmp_path / "evidence",
        model_root=tmp_path / "models",
        database=tmp_path / "runtime" / "knowledge.sqlite3",
        structure_root=tmp_path / "runtime" / "structure",
    ).ensure()


def _make_pdf(path: Path, pages: int = 2) -> None:
    import fitz

    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page(width=420, height=600)
        page.insert_text(
            (40, 60),
            f"Page {index + 1}: CT showed a 5 mm pulmonary nodule.",
        )
        page.draw_rect(fitz.Rect(40, 100, 180, 220))
        page.insert_text((55, 145), "Original figure")
    doc.save(path)
    doc.close()


class TranslationProductV2Tests(unittest.TestCase):
    def test_intelligent_translation_is_primary_for_medical_chinese(self):
        with tempfile.TemporaryDirectory() as temp:
            llm = _SmartLLM()
            engine = MultiModelTranslationEngine(_paths(Path(temp)), llm)

            decision = engine.translate(
                "CT showed a 5 mm pulmonary nodule.",
                "中文",
                smart_level="smart2",
            )
            self.assertEqual(decision.text, "CT显示5 mm肺结节，边缘清楚。")
            self.assertIn("qwen35", decision.backend)
            self.assertEqual(llm.profiles[-1], "translation")
            self.assertEqual(llm.max_tokens[-1], 512)

            with self.assertRaises(RuntimeError):
                engine.translate(
                    "CT showed a 5 mm pulmonary nodule.",
                    "中文",
                    smart_level="smart1",
                )

    def test_translation_chunk_size_is_bounded_and_configurable(self):
        with patch.dict(
            os.environ,
            {"PHOENIX_TRANSLATION_CHUNK_CHARS": "100"},
        ):
            self.assertEqual(_translation_chunk_chars(), 1600)
        with patch.dict(
            os.environ,
            {"PHOENIX_TRANSLATION_CHUNK_CHARS": "999999"},
        ):
            self.assertEqual(_translation_chunk_chars(), 6000)

    def test_original_page_above_translation_pdf_and_split_volumes(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            _make_pdf(source, pages=2)
            pages_root = root / "pages"
            pages_root.mkdir()
            (pages_root / "000001.txt").write_text(
                "第一页中文医学译文。",
                encoding="utf-8",
            )
            (pages_root / "000002.txt").write_text(
                "第二页中文医学译文。",
                encoding="utf-8",
            )

            complete, parts = TranslationPDFBuilder(
                source,
                pages_root,
                root / "outputs",
            ).build(
                start_page=1,
                total_pages=2,
                layout=LAYOUT_ORIGINAL_BILINGUAL,
                part_pages=1,
            )

            self.assertTrue(complete.is_file())
            self.assertEqual(len(parts), 2)
            self.assertTrue(all(path.is_file() for path in parts))

            source_doc = fitz.open(source)
            translated_doc = fitz.open(complete)
            try:
                self.assertEqual(translated_doc.page_count, 2)
                self.assertGreater(
                    translated_doc[0].rect.height,
                    source_doc[0].rect.height,
                )
            finally:
                source_doc.close()
                translated_doc.close()

    def test_translation_pause_and_resume_keeps_completed_pages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "book.pdf"
            _make_pdf(source, pages=2)
            translator = PDFTranslator(_paths(root), _UnavailableLLM())

            backend = object()
            translator.engine.active_backends = (
                lambda target_language="中文", smart_level="smart1": [backend]
            )
            translator.engine.available_backends = lambda: ["internal_test"]
            translator.engine.unload = lambda: None

            def fake_translate(source_text, target_language="中文", *, smart_level="smart1"):
                text = f"译文：{source_text.strip()}"
                quality = QualityReport(True, 1.0, ())
                attempt = TranslationAttempt(
                    backend="internal_test",
                    text=text,
                    quality=quality,
                )
                return TranslationDecision(
                    text=text,
                    backend="internal_test",
                    quality=quality,
                    needs_review=False,
                    attempts=(attempt,),
                )

            translator.engine.translate = fake_translate

            pause_calls = {"count": 0}

            def pause_after_first_page():
                pause_calls["count"] += 1
                return pause_calls["count"] >= 2

            first = translator.translate_book(
                source,
                target_language="中文",
                smart_level="smart1",
                export_format=EXPORT_TXT,
                should_pause=pause_after_first_page,
                page_preview=lambda page, text, path: None,
            )
            self.assertTrue(first.paused)
            self.assertEqual(first.pages_done, 1)
            self.assertEqual(first.smart_level, "smart2")

            previews = []
            second = translator.translate_book(
                source,
                target_language="中文",
                smart_level="smart1",
                export_format=EXPORT_TXT,
                should_pause=lambda: False,
                page_preview=lambda page, text, path: previews.append(page),
            )
            self.assertFalse(second.paused)
            self.assertGreaterEqual(second.resumed_pages, 1)
            self.assertTrue(second.output_path.is_file())
            text = second.output_path.read_text(encoding="utf-8")
            self.assertIn("第 1 页", text)
            self.assertIn("第 2 页", text)
            self.assertEqual(previews, [1, 2])


if __name__ == "__main__":
    unittest.main()
