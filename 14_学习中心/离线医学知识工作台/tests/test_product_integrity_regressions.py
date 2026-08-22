from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.llm_safe import LocalLLM
from phoenix_knowledge.output_contracts import OutputContractError
from phoenix_knowledge.pdf_assets import PDFAssetStore, markdown_images
from phoenix_knowledge.pdf_parser import iter_pdf_pages
from phoenix_knowledge.product_document_ingest import ProductDocumentIngestor
from phoenix_knowledge.rich_export import MultiFormatExporter
from phoenix_knowledge.translation_models import TranslationValidator
from phoenix_knowledge.workbench import MedicalKnowledgeWorkbench


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


def _png_bytes() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )
    header = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
    raw = b"".join([b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4)])
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def _scan_pdf(path: Path) -> None:
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_image(fitz.Rect(20, 20, 280, 280), stream=_png_bytes())
    doc.save(path)
    doc.close()


def _text_pdf(path: Path, text: str) -> None:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class ProductIntegrityRegressionTests(unittest.TestCase):
    def test_translation_validator_rejects_changed_sign_and_unit(self):
        validator = TranslationValidator()
        sign = validator.validate(
            "CT attenuation measured -20 HU in the lesion.",
            "病灶CT衰减值为20 HU。",
        )
        unit = validator.validate(
            "The pulmonary nodule measured 12 mm.",
            "肺结节大小为12 cm。",
        )
        self.assertFalse(sign.ok)
        self.assertFalse(unit.ok)
        self.assertTrue(any("正负号" in reason for reason in sign.reasons))
        self.assertTrue(any("单位" in reason for reason in unit.reasons))

    def test_asset_cache_isolated_by_full_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_dir = root / "A"
            second_dir = root / "B"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "Oncology.pdf"
            second = second_dir / "Oncology.pdf"
            _text_pdf(first, "first")
            _text_pdf(second, "second")
            store = PDFAssetStore(root / "runtime")
            self.assertNotEqual(store.document_root(first), store.document_root(second))

    def test_pdf_asset_manifest_invalidates_when_source_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "Oncology.pdf"
            _text_pdf(pdf, "version one")
            store = PDFAssetStore(root / "runtime")
            first = store.extract(pdf)
            old_sha = first["source_sha256"]
            pdf.unlink()
            _text_pdf(pdf, "version two changed")
            second = store.extract(pdf)
            self.assertNotEqual(old_sha, second["source_sha256"])
            self.assertEqual(second["source_path"], str(pdf.resolve()))

    def test_scanned_pdf_translation_iterator_uses_same_local_ocr_path(self):
        with tempfile.TemporaryDirectory() as temp:
            pdf = Path(temp) / "scan.pdf"
            _scan_pdf(pdf)
            with patch(
                "phoenix_knowledge.pdf_parser._ocr_page_text",
                return_value="OCR识别成功：右肺结节12 mm",
            ):
                pages = list(iter_pdf_pages(pdf))
            self.assertEqual(len(pages), 1)
            self.assertIn("肺结节12 mm", pages[0][1])

    def test_parenthesized_image_path_and_chinese_pdf_caption_survive_export(self):
        import fitz
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "topic_assets"
            assets.mkdir()
            image = assets / "Lung (CT).png"
            image.write_bytes(_png_bytes())
            source = root / "topic.md"
            generated = markdown_images(
                [image],
                relative_to=root,
                label="肺结节 第12页",
            )
            self.assertIn("(<topic_assets/Lung (CT).png>)", generated)
            source.write_text(
                "# 肺结节\n\n" + generated + "\n",
                encoding="utf-8",
            )
            bundle = MultiFormatExporter(root / "out").export_path(
                source,
                title="测试专题",
            )
            with zipfile.ZipFile(bundle.docx) as zf:
                self.assertTrue(
                    any(name.startswith("word/media/") for name in zf.namelist())
                )
            pdf = fitz.open(bundle.pdf)
            try:
                self.assertTrue(any(page.get_images(full=True) for page in pdf))
                text = "\n".join(page.get_text("text") for page in pdf)
                self.assertIn("图：肺结节 第12页", text)
            finally:
                pdf.close()

    def test_reimport_same_new_version_reuses_existing_suffixed_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            db = KnowledgeDB(paths.database)
            try:
                ingestor = ProductDocumentIngestor(db, paths)
                external = root / "external"
                external.mkdir()
                source = external / "book.txt"
                source.write_text("version one", encoding="utf-8")
                first = ingestor._library_copy(source)
                self.assertEqual(first.name, "book.txt")
                source.write_text("version two", encoding="utf-8")
                second = ingestor._library_copy(source)
                self.assertEqual(second.name, "book_2.txt")
                third = ingestor._library_copy(source)
                self.assertEqual(third, second)
                self.assertFalse((paths.source_root / "book_3.txt").exists())
            finally:
                db.close()

    def test_export_failure_blocks_completion_instead_of_silent_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workbench = MedicalKnowledgeWorkbench(_paths(root))
            try:
                source = root / "organized.md"
                source.write_text("# result\n", encoding="utf-8")
                task_id = workbench.db.create_task(
                    "deep_organize",
                    {"title": "topic", "instruction": "instruction", "chunk_ids": []},
                    total=1,
                )
                workbench.organizer = SimpleNamespace(
                    organize=lambda title, instruction, **kwargs: (source, task_id)
                )
                with patch(
                    "phoenix_knowledge.workbench_stability_core.transactional_export_path",
                    side_effect=RuntimeError("simulated export failure"),
                ):
                    with self.assertRaises(OutputContractError):
                        workbench.organize("topic", "instruction")
                self.assertIn("simulated export failure", workbench.last_export_error)
                self.assertIsNone(workbench.last_export_bundle)
                task = workbench.db.get_task(task_id)
                self.assertEqual(str(task["status"]), "failed")
            finally:
                workbench.close()

    def test_http_waits_are_bounded_and_configurable(self):
        with patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_LOCAL_TIMEOUT": "9999",
                "PHOENIX_KNOWLEDGE_REMOTE_TIMEOUT": "30",
            },
            clear=False,
        ):
            self.assertEqual(
                LocalLLM._timeout("PHOENIX_KNOWLEDGE_LOCAL_TIMEOUT", 180),
                600,
            )
            self.assertEqual(
                LocalLLM._timeout("PHOENIX_KNOWLEDGE_REMOTE_TIMEOUT", 180),
                30,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
