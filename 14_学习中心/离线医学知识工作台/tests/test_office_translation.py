from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.office_translation import (
    OfficeDocumentTranslator,
    OfficeTranslationError,
    validate_office_package,
)
from phoenix_knowledge.translation_models import (
    MultiModelTranslationEngine,
    QualityReport,
    TranslationAttempt,
    TranslationDecision,
)
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
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _write_pptx(path: Path, *, acronym_labels: bool = False) -> None:
    types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""
    presentation = """<?xml version="1.0" encoding="UTF-8"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"""
    slide1 = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <p:cSld><p:spTree><p:sp><p:txBody><a:p>
  <a:r><a:t>Pulmonary nodule</a:t></a:r>
  <a:r><a:t>No pleural effusion</a:t></a:r>
 </a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"""
    slide2 = slide1.replace(
        "Pulmonary nodule", "Right lung lesion 12 mm"
    ).replace("No pleural effusion", "CT follow-up")
    if acronym_labels:
        slide1 = slide1.replace("Pulmonary nodule", "DWI")
        slide2 = slide2.replace("Right lung lesion 12 mm", "ADC")
    slide_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("ppt/presentation.xml", presentation)
        archive.writestr("ppt/slides/slide1.xml", slide1)
        archive.writestr("ppt/slides/slide2.xml", slide2)
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", slide_rels)
        archive.writestr("ppt/media/image1.png", _png_bytes())


def _write_docx(path: Path) -> None:
    types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
</Types>"""
    document = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:body>
  <w:p><w:r><w:t>Sensitivity was 91%.</w:t></w:r></w:p>
  <w:p><w:r><w:t>No metastasis.</w:t></w:r></w:p>
 </w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("word/document.xml", document)


class _Engine:
    translations = {
        "Pulmonary nodule": "肺结节",
        "No pleural effusion": "未见胸腔积液",
        "Right lung lesion 12 mm": "右肺病灶 12 mm",
        "CT follow-up": "CT 随访",
        "Sensitivity was 91%.": "敏感度为 91%。",
        "No metastasis.": "未见转移。",
    }

    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def active_backends(self, target_language="中文", smart_level="smart2"):
        return [self]

    def available_backends(self):
        return ["quality_test"]

    def translate_segments(self, sources, target_language="中文", *, smart_level="smart2"):
        self.calls.append(tuple(sources))
        decisions = []
        for source in sources:
            text = self.translations[source]
            quality = QualityReport(True, 1.0, ())
            attempt = TranslationAttempt("quality_test", text, quality)
            decisions.append(
                TranslationDecision(
                    text=text,
                    backend="quality_test",
                    quality=quality,
                    needs_review=False,
                    attempts=(attempt,),
                )
            )
        return tuple(decisions)

    def unload(self):
        pass


class _BatchLLM:
    def __init__(self):
        self.profiles: list[str | None] = []

    def available(self, profile=None):
        return True

    def generate(self, prompt, max_new_tokens=1200, *, profile=None):
        self.profiles.append(profile)
        if "医学翻译纠错器" in prompt:
            return "未见胸腔积液。"
        return json.dumps(
            [
                {"id": "S0001", "translation": "CT显示5 mm肺结节。"},
                {"id": "S0002", "translation": "Pleural effusion."},
            ],
            ensure_ascii=False,
        )


class _FailEngine:
    def active_backends(self, target_language="中文", smart_level="smart2"):
        return [self]

    def formal_backend_names(self, target_language="中文"):
        return ["quality_test"]

    def translate_segments(self, *args, **kwargs):
        raise RuntimeError("batch unavailable")

    def translate(self, *args, **kwargs):
        raise RuntimeError("item unavailable")

    def unload(self):
        pass


class OfficeTranslationTests(unittest.TestCase):
    def test_pptx_stays_pptx_preserves_media_and_previews_each_slide(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "lecture.pptx"
            _write_pptx(source)
            engine = _Engine()
            translator = OfficeDocumentTranslator(_paths(root), engine)
            previews: list[tuple[int, str]] = []

            result = translator.translate_document(
                source,
                page_preview=lambda unit, text, path: previews.append((unit, text)),
                smart_level="smart1",
                medical_quality_required=False,
            )

            self.assertEqual(result.output_path.suffix, ".pptx")
            self.assertEqual(result.output_paths, (result.output_path,))
            self.assertEqual(result.smart_level, "smart2")
            self.assertEqual(len(previews), 2)
            self.assertEqual(len(engine.calls), 2)
            report = validate_office_package(source, result.output_path)
            self.assertTrue(report["media_preserved"])
            self.assertEqual(report["media_count"], 1)
            with zipfile.ZipFile(source) as before, zipfile.ZipFile(result.output_path) as after:
                self.assertEqual(
                    before.read("ppt/media/image1.png"),
                    after.read("ppt/media/image1.png"),
                )
                translated = after.read("ppt/slides/slide1.xml").decode("utf-8")
                self.assertIn("肺结节", translated)
                self.assertIn("未见胸腔积液", translated)

            resumed = translator.translate_document(source)
            self.assertEqual(resumed.resumed_pages, 2)
            self.assertEqual(len(engine.calls), 2)

    def test_pptx_pure_medical_acronyms_use_cached_deck_glossary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "acronyms.pptx"
            _write_pptx(source, acronym_labels=True)
            engine = _Engine()
            translator = OfficeDocumentTranslator(_paths(root), engine)

            result = translator.translate_document(source)

            with zipfile.ZipFile(result.output_path) as archive:
                slide1 = archive.read("ppt/slides/slide1.xml").decode("utf-8")
                slide2 = archive.read("ppt/slides/slide2.xml").decode("utf-8")
            self.assertIn("弥散加权成像（DWI）", slide1)
            self.assertIn("表观弥散系数（ADC）", slide2)
            sent_to_model = {source for call in engine.calls for source in call}
            self.assertNotIn("DWI", sent_to_model)
            self.assertNotIn("ADC", sent_to_model)

            glossary_files = list(translator.output_root.rglob("医学缩写术语表.json"))
            self.assertEqual(len(glossary_files), 1)
            glossary = json.loads(glossary_files[0].read_text(encoding="utf-8"))
            self.assertEqual(glossary["glossary"]["DWI"]["chinese"], "弥散加权成像")
            self.assertEqual(glossary["glossary"]["ADC"]["chinese"], "表观弥散系数")

    def test_docx_paper_stays_docx_and_translates_one_paragraph_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "paper.docx"
            _write_docx(source)
            engine = _Engine()
            result = OfficeDocumentTranslator(_paths(root), engine).translate_document(source)
            self.assertEqual(result.output_path.suffix, ".docx")
            self.assertEqual(len(engine.calls), 1)
            with zipfile.ZipFile(result.output_path) as archive:
                xml = archive.read("word/document.xml").decode("utf-8")
            self.assertIn("敏感度为 91%", xml)
            self.assertIn("未见转移", xml)

    def test_public_workbench_dispatches_office_without_pdf_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            source = root / "paper.docx"
            _write_docx(source)
            workbench = MedicalKnowledgeWorkbench(paths)
            try:
                workbench.office_translator = OfficeDocumentTranslator(
                    paths,
                    _Engine(),
                )
                result = workbench.translate_book(source)
                self.assertEqual(result.output_path.suffix, ".docx")
                self.assertTrue(result.output_path.is_file())
                self.assertEqual(workbench.status()["office_translation_contract"], 3)
            finally:
                workbench.close()

    def test_batch_quality_model_retries_only_failed_segment_without_reasoning(self):
        with tempfile.TemporaryDirectory() as temp:
            llm = _BatchLLM()
            engine = MultiModelTranslationEngine(_paths(Path(temp)), llm)
            decisions = engine.translate_segments(
                [
                    "CT showed a 5 mm pulmonary nodule.",
                    "No pleural effusion.",
                ],
                "中文",
            )
            self.assertTrue(all(item.quality.ok for item in decisions))
            self.assertEqual(llm.profiles, ["translation", "translation"])
            self.assertIn("未见", decisions[1].text)

    def test_office_resume_keeps_original_requested_start_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "lecture.pptx"
            _write_pptx(source)
            engine = _Engine()
            translator = OfficeDocumentTranslator(_paths(root), engine)
            pause_checks = iter((False, True))

            paused = translator.translate_document(
                source,
                start_page=2,
                should_pause=lambda: next(pause_checks, True),
            )
            self.assertTrue(paused.paused)
            self.assertEqual(paused.start_page, 2)

            resumed = translator.translate_document(source)
            self.assertFalse(resumed.paused)
            self.assertEqual(resumed.start_page, 2)
            self.assertEqual(resumed.resumed_pages, 1)
            self.assertEqual(len(engine.calls), 1)
            with zipfile.ZipFile(resumed.output_path) as archive:
                slide1 = archive.read("ppt/slides/slide1.xml").decode("utf-8")
                slide2 = archive.read("ppt/slides/slide2.xml").decode("utf-8")
            self.assertIn("Pulmonary nodule", slide1)
            self.assertIn("右肺病灶 12 mm", slide2)

    def test_all_failed_office_translation_keeps_checkpoints_but_blocks_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "paper.docx"
            _write_docx(source)
            translator = OfficeDocumentTranslator(_paths(root), _FailEngine())
            with self.assertRaises(OfficeTranslationError):
                translator.translate_document(source)
            self.assertFalse(any(translator.output_root.rglob("*译本.docx")))
            self.assertTrue(any(translator.output_root.rglob("checkpoint.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
