from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.translation_output_validation import (
    TranslationOutputError,
    assert_stable_against_previous,
    build_input_signature,
    validate_pdf,
    write_integrity_report,
)
from phoenix_knowledge.translation_pdf import TranslationPDFBuilder
from phoenix_knowledge.translation_stability_core import (
    LAYOUT_SOURCE_TRANSLATED,
)
from phoenix_knowledge.translator import PDFTranslator


class TranslationOutputStabilityTests(unittest.TestCase):
    @staticmethod
    def _make_source(path: Path, pages: int = 3) -> None:
        import fitz

        doc = fitz.open()
        try:
            for index in range(pages):
                page = doc.new_page(width=595, height=842)
                page.insert_textbox(
                    fitz.Rect(48, 48, 547, 160),
                    (
                        f"Chapter {index + 1}. CT demonstrates no pleural effusion. "
                        "A 12 mm lesion is present in the right kidney."
                    ),
                    fontsize=11,
                )
                page.insert_text((290, 815), str(index + 1), fontsize=8)
            doc.save(path, garbage=2, deflate=True, use_objstms=1)
        finally:
            doc.close()

    @staticmethod
    def _write_pages(root: Path, pages: int) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for page_number in range(1, pages + 1):
            (root / f"{page_number:06d}.txt").write_text(
                f"第{page_number}章。CT显示无胸腔积液。右肾可见12 mm病变。",
                encoding="utf-8",
            )

    def test_runtime_translation_patch_stack_is_collapsed(self):
        from phoenix_knowledge.translation_output_validation import CONTRACT_VERSION

        self.assertEqual(
            getattr(PDFTranslator, "_phoenix_translation_wrapper_depth", None),
            1,
        )
        self.assertEqual(
            getattr(PDFTranslator, "_phoenix_stability_contract", None),
            CONTRACT_VERSION,
        )
        self.assertEqual(
            getattr(TranslationPDFBuilder, "_phoenix_stability_contract", None),
            CONTRACT_VERSION,
        )
        self.assertEqual(
            PDFTranslator.translate_book.__module__,
            "phoenix_knowledge.translation_stability_core",
        )
        self.assertEqual(
            TranslationPDFBuilder.build.__module__,
            "phoenix_knowledge.translation_stability_core",
        )

    def test_truncated_pdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            self._make_source(source, pages=2)
            self._write_pages(pages_root, pages=2)
            damaged = root / "damaged.pdf"
            payload = source.read_bytes()
            damaged.write_bytes(payload[: max(32, len(payload) // 2)])

            with self.assertRaises(TranslationOutputError):
                validate_pdf(
                    damaged,
                    expected_pages=2,
                    pages_root=pages_root,
                    start_page=1,
                )

    def test_same_input_structure_change_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            self._make_source(source, pages=2)
            self._write_pages(pages_root, pages=2)
            signature = build_input_signature(
                source_pdf=source,
                pages_root=pages_root,
                start_page=1,
                total_pages=2,
                layout=LAYOUT_SOURCE_TRANSLATED,
            )
            report_path = root / "PDF完整性报告.json"
            write_integrity_report(
                report_path,
                signature=signature,
                pdf_report={
                    "structure_sha256": "a" * 64,
                    "passed": True,
                },
            )
            with self.assertRaises(TranslationOutputError):
                assert_stable_against_previous(
                    report_path,
                    signature=signature,
                    current_structure_sha256="b" * 64,
                )

    def test_newer_page_than_audit_is_invalidated_as_interrupted_checkpoint(self):
        from phoenix_knowledge.translation_stability_core import (
            _invalidate_unstable_resume_pages,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            audit_root = root / "audit"
            pages_root.mkdir()
            audit_root.mkdir()
            self._make_source(source, pages=1)

            audit_file = audit_root / "000001.json"
            audit_file.write_text(
                json.dumps(
                    {
                        "warning_count": 0,
                        "parts": [{"backend": "storage_test"}],
                    }
                ),
                encoding="utf-8",
            )
            page_file = pages_root / "000001.txt"
            page_file.write_text("半截但仍是合法UTF-8的译文", encoding="utf-8")
            os.utime(audit_file, ns=(1_000_000_000, 1_000_000_000))
            os.utime(page_file, ns=(2_000_000_000, 2_000_000_000))

            class _Translator:
                def _book_paths(self, *_args, **_kwargs):
                    return (
                        root,
                        pages_root,
                        audit_root,
                        root / "checkpoint.json",
                        root / "完整译文.txt",
                    )

                @staticmethod
                def _read_json(path):
                    return json.loads(Path(path).read_text(encoding="utf-8"))

            removed = _invalidate_unstable_resume_pages(
                _Translator(),
                source,
                "中文",
                retry_warning_pages=False,
            )
            self.assertEqual(removed, 1)
            self.assertFalse(page_file.exists())

    def test_low_disk_space_fails_before_staging_or_overwriting_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            output_root = root / "output"
            self._make_source(source, pages=2)
            self._write_pages(pages_root, pages=2)
            builder = TranslationPDFBuilder(source, pages_root, output_root)

            class _Usage:
                free = 1

            with patch(
                "phoenix_knowledge.translation_stability_core.shutil.disk_usage",
                return_value=_Usage(),
            ):
                with self.assertRaises(TranslationOutputError):
                    builder.build(
                        start_page=1,
                        total_pages=2,
                        layout=LAYOUT_SOURCE_TRANSLATED,
                        part_pages=0,
                    )

            self.assertEqual(list(output_root.parent.glob(".pxpdf-*")), [])
            self.assertFalse(any(output_root.glob("*.pdf")))

    def test_failed_stage_is_cleaned_without_publishing_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            output_root = root / "output"
            self._make_source(source, pages=2)
            self._write_pages(pages_root, pages=2)
            builder = TranslationPDFBuilder(source, pages_root, output_root)

            def explode(staged_builder, **kwargs):
                staged_builder.output_root.mkdir(parents=True, exist_ok=True)
                (staged_builder.output_root / "partial.bin").write_bytes(b"x" * 4096)
                raise RuntimeError("synthetic build failure")

            with patch(
                "phoenix_knowledge.translation_layout_compact._build_source_translated",
                side_effect=explode,
            ):
                with self.assertRaises(RuntimeError):
                    builder.build(
                        start_page=1,
                        total_pages=2,
                        layout=LAYOUT_SOURCE_TRANSLATED,
                        part_pages=0,
                    )

            leftovers = list(output_root.parent.glob(".pxpdf-*"))
            self.assertEqual(leftovers, [])
            self.assertFalse(any(output_root.glob("*.pdf")))

    def test_failed_rebuild_does_not_overwrite_last_good_pdf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            pages_root = root / "pages"
            output_root = root / "output"
            self._make_source(source, pages=3)
            self._write_pages(pages_root, pages=3)

            builder = TranslationPDFBuilder(source, pages_root, output_root)
            complete, parts = builder.build(
                start_page=1,
                total_pages=3,
                layout=LAYOUT_SOURCE_TRANSLATED,
                part_pages=0,
            )
            self.assertEqual(parts, ())
            before = hashlib.sha256(complete.read_bytes()).hexdigest()

            integrity_path = output_root / "PDF完整性报告.json"
            payload = json.loads(integrity_path.read_text(encoding="utf-8"))
            payload["structure_sha256"] = "0" * 64
            integrity_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self.assertRaises(TranslationOutputError):
                builder.build(
                    start_page=1,
                    total_pages=3,
                    layout=LAYOUT_SOURCE_TRANSLATED,
                    part_pages=0,
                )

            after = hashlib.sha256(complete.read_bytes()).hexdigest()
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
