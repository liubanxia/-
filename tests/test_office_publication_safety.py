from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from phoenix_knowledge import bootstrap_runtime

# This file validates the formal production contract, not the raw helper class.
bootstrap_runtime()

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.office_translation import OfficeDocumentTranslator, OfficeTranslationError
from phoenix_knowledge.translation_models import QualityReport, TranslationAttempt, TranslationDecision


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


def _write_docx(path: Path, text: str) -> None:
    types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="xml" ContentType="application/xml"/>
</Types>"""
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("word/document.xml", document)


class _DecisionEngine:
    def __init__(self, decision: TranslationDecision):
        self.decision = decision

    def active_backends(self, target_language="中文", smart_level="smart2"):
        return [self]

    def formal_backend_names(self, target_language="中文"):
        return ["production_test"]

    def translate_segments(self, sources, target_language="中文", *, smart_level="smart2"):
        return tuple(self.decision for _ in sources)

    def translate(self, *args, **kwargs):
        return self.decision

    def unload(self):
        pass


def _decision(text: str, *, ok: bool, review: bool, backend: str = "production_test"):
    quality = QualityReport(ok, 0.95 if ok else 0.1, () if ok else ("quality failed",))
    attempt = TranslationAttempt(backend, text, quality)
    return TranslationDecision(
        text=text,
        backend=backend,
        quality=quality,
        needs_review=review,
        attempts=(attempt,),
    )


class OfficePublicationSafetyTest(unittest.TestCase):
    def _audit_rows(self, translator: OfficeDocumentTranslator) -> list[dict]:
        files = list(translator.output_root.rglob("units/*.json"))
        self.assertTrue(files)
        rows: list[dict] = []
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("translations") or [])
        return rows

    def test_review_required_candidate_is_never_written_to_formal_docx(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "review.docx"
            original = "Pulmonary nodule present."
            rejected = "候选译文但质量门未通过。"
            _write_docx(source, original)
            translator = OfficeDocumentTranslator(
                _paths(root),
                _DecisionEngine(_decision(rejected, ok=False, review=True)),
            )

            try:
                result = translator.translate_document(source)
            except OfficeTranslationError:
                # Blocking the whole formal output is also safe when there is no
                # accepted segment at all.
                self.assertFalse(any(translator.output_root.rglob("*译本.docx")))
            else:
                with zipfile.ZipFile(result.output_path) as archive:
                    xml = archive.read("word/document.xml").decode("utf-8")
                self.assertIn(original, xml)
                self.assertNotIn(rejected, xml)

            rows = self._audit_rows(translator)
            self.assertTrue(any(row.get("rejected_candidate") == rejected for row in rows))
            self.assertTrue(all(not row.get("publication_approved", False) for row in rows))

    def test_model_refusal_is_never_written_to_formal_docx(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "refusal.docx"
            original = "No pleural effusion."
            refusal = "抱歉，我无法直接处理医疗或健康领域内容。"
            _write_docx(source, original)
            translator = OfficeDocumentTranslator(
                _paths(root),
                _DecisionEngine(_decision(refusal, ok=True, review=False, backend="api_teacher")),
            )

            try:
                result = translator.translate_document(source)
            except OfficeTranslationError:
                self.assertFalse(any(translator.output_root.rglob("*译本.docx")))
            else:
                with zipfile.ZipFile(result.output_path) as archive:
                    xml = archive.read("word/document.xml").decode("utf-8")
                self.assertIn(original, xml)
                self.assertNotIn(refusal, xml)

            rows = self._audit_rows(translator)
            self.assertTrue(any(row.get("refused_output") for row in rows))

    def test_accepted_translation_still_replaces_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "accepted.docx"
            original = "No pleural effusion."
            accepted = "未见胸腔积液。"
            _write_docx(source, original)
            translator = OfficeDocumentTranslator(
                _paths(root),
                _DecisionEngine(_decision(accepted, ok=True, review=False)),
            )

            result = translator.translate_document(source)
            with zipfile.ZipFile(result.output_path) as archive:
                xml = archive.read("word/document.xml").decode("utf-8")
            self.assertIn(accepted, xml)
            self.assertNotIn(original, xml)

            rows = self._audit_rows(translator)
            self.assertTrue(all(row.get("publication_approved", False) for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
