from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from phoenix_knowledge.answerer import _locator as answer_locator
from phoenix_knowledge.cnki_converter import CNKI_EXTENSIONS
from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.product_document_ingest import (
    ProductDocumentIngestor,
    SUPPORTED_EXTENSIONS,
)
from phoenix_knowledge.retrieval import Evidence
from phoenix_knowledge.scholarly_ingest import (
    DIRECT_SCHOLARLY_EXTENSIONS,
    ScholarlyParser,
)


def _paths(root: Path) -> WorkbenchPaths:
    return WorkbenchPaths(
        project_root=root,
        source_root=root / "library",
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

    header = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    raw = b"".join([b"\x00" + b"\xff\x00\x00" * 2 for _ in range(2)])
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _pdf(path: Path, text: str = "Pulmonary nodule 12 mm") -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class ScholarlyFormatTests(unittest.TestCase):
    def test_all_declared_research_formats_are_first_class_inputs(self):
        expected = {
            ".html", ".htm", ".xml", ".nxml", ".jats",
            ".nbib", ".ris", ".bib", ".bibtex", ".json", ".csljson",
            ".caj", ".nh", ".hn", ".kdh", ".teb", ".c8",
        }
        self.assertTrue(expected.issubset(SUPPORTED_EXTENSIONS))
        self.assertTrue(DIRECT_SCHOLARLY_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS))
        self.assertTrue(CNKI_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS))

    def test_jats_extracts_metadata_sections_and_local_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fig1.png").write_bytes(_png_bytes())
            source = root / "paper.nxml"
            source.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">
 <front>
  <journal-meta><journal-title-group><journal-title>Radiology Test</journal-title></journal-title-group></journal-meta>
  <article-meta>
   <article-id pub-id-type="doi">10.1234/test.2026.001</article-id>
   <article-id pub-id-type="pmid">12345678</article-id>
   <title-group><article-title>CT Pulmonary Nodule Study</article-title></title-group>
   <contrib-group><contrib contrib-type="author"><name><surname>Li</surname><given-names>Ming</given-names></name></contrib></contrib-group>
   <pub-date><year>2026</year></pub-date>
   <abstract><p>We studied 120 CT examinations.</p></abstract>
   <kwd-group><kwd>CT</kwd><kwd>pulmonary nodule</kwd></kwd-group>
  </article-meta>
 </front>
 <body>
  <sec><title>Methods</title><p>Slice thickness was 1 mm.</p>
   <fig><label>Fig. 1</label><caption><p>Nodule example.</p></caption><graphic xlink:href="fig1.png"/></fig>
  </sec>
  <sec><title>Results</title><p>AUC was 0.91.</p></sec>
 </body>
</article>""",
                encoding="utf-8",
            )
            parser = ScholarlyParser(root / "runtime")
            parsed = parser.parse(source)
            self.assertEqual(parsed.primary_title, "CT Pulmonary Nodule Study")
            self.assertEqual(parsed.records[0].doi, "10.1234/test.2026.001")
            self.assertEqual(parsed.records[0].pmid, "12345678")
            joined = "\n".join(unit.text for unit in parsed.units)
            self.assertIn("Slice thickness was 1 mm", joined)
            self.assertIn("AUC was 0.91", joined)
            self.assertTrue(parsed.images)

    def test_html_citation_meta_and_body_are_indexed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "paper.html"
            source.write_text(
                """<html><head>
<meta name="citation_title" content="MRI Liver Study">
<meta name="citation_author" content="Alice Zhang">
<meta name="citation_journal_title" content="Medical Imaging">
<meta name="citation_doi" content="10.5555/mri.1">
<meta name="citation_publication_date" content="2026-05-01">
<meta name="description" content="Prospective MRI study.">
</head><body><h1>MRI Liver Study</h1><h2>Methods</h2>
<p>Fifty patients underwent dynamic contrast MRI.</p></body></html>""",
                encoding="utf-8",
            )
            parsed = ScholarlyParser(root / "runtime").parse(source)
            record = parsed.records[0]
            self.assertEqual(record.title, "MRI Liver Study")
            self.assertEqual(record.doi, "10.5555/mri.1")
            self.assertEqual(record.year, "2026")
            self.assertIn("Fifty patients", parsed.units[0].text)

    def test_nbib_ris_bibtex_and_csl_json_are_searchable_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixtures = {
                "paper.nbib": (
                    "PMID- 11111111\n"
                    "TI  - CT diagnosis of appendicitis\n"
                    "FAU - Zhang, Wei\n"
                    "JT  - Journal of CT\n"
                    "DP  - 2025 Jan\n"
                    "AID - 10.1000/appendix.1 [doi]\n"
                    "AB  - Sensitivity was 94 percent.\n"
                ),
                "paper.ris": (
                    "TY  - JOUR\n"
                    "TI  - MRI diagnosis of liver tumor\n"
                    "AU  - Li, Ming\n"
                    "JO  - Imaging Journal\n"
                    "PY  - 2024\n"
                    "DO  - 10.1000/liver.2\n"
                    "AB  - AUC was 0.93.\n"
                    "ER  -\n"
                ),
                "paper.bib": (
                    "@article{kidney2023,\n"
                    " title={CT detection of renal stones},\n"
                    " author={Chen, Li and Wang, Bo},\n"
                    " journal={Radiology Research},\n"
                    " year={2023},\n"
                    " doi={10.1000/stone.3},\n"
                    " abstract={Specificity was 96 percent.}\n"
                    "}\n"
                ),
            }
            for name, text in fixtures.items():
                (root / name).write_text(text, encoding="utf-8")
            (root / "paper.json").write_text(
                json.dumps(
                    {
                        "type": "article-journal",
                        "title": "Chest radiograph fracture detection",
                        "author": [{"given": "Lei", "family": "Wu"}],
                        "container-title": "Medical AI",
                        "issued": {"date-parts": [[2026, 1, 2]]},
                        "DOI": "10.1000/fracture.4",
                        "abstract": "External validation included 800 exams.",
                    }
                ),
                encoding="utf-8",
            )

            parser = ScholarlyParser(root / "runtime")
            for name in ["paper.nbib", "paper.ris", "paper.bib", "paper.json"]:
                parsed = parser.parse(root / name)
                self.assertTrue(parsed.records)
                self.assertTrue(parsed.units)
                self.assertTrue(parsed.records[0].title)

    def test_identifier_catalog_binds_fulltext_and_citation_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            xml = root / "fulltext.nxml"
            xml.write_text(
                """<article><front><article-meta>
<article-id pub-id-type="doi">10.7777/bind.1</article-id>
<title-group><article-title>Binding Study</article-title></title-group>
</article-meta></front><body><sec><title>Results</title><p>AUC 0.90.</p></sec></body></article>""",
                encoding="utf-8",
            )
            ris = root / "citation.ris"
            ris.write_text(
                "TY  - JOUR\nTI  - Binding Study\nDO  - 10.7777/bind.1\nER  -\n",
                encoding="utf-8",
            )
            parser = ScholarlyParser(root / "runtime")
            full = parser.parse(xml)
            cite = parser.parse(ris)
            parser.catalog.register(xml, full.records)
            parser.catalog.register(ris, cite.records)
            data = json.loads(parser.catalog.path.read_text(encoding="utf-8"))
            bucket = data["records"]["doi:10.7777/bind.1"]
            self.assertEqual(len(bucket), 2)

    def test_cnki_original_path_is_preserved_after_offline_pdf_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = _paths(root)
            source = root / "downloaded.caj"
            source.write_bytes(b"CAJ placeholder")
            converted = root / "converted.pdf"
            _pdf(converted, "Renal stone 6 mm")
            db = KnowledgeDB(paths.database)
            try:
                ingestor = ProductDocumentIngestor(db, paths)
                with patch.object(ingestor.cnki, "convert", return_value=converted):
                    result = ingestor.ingest(
                        source,
                        copy_into_library=False,
                        extract_images=False,
                    )
                row = db.get_document(result.document_id)
                self.assertEqual(Path(row["path"]), source.resolve())
                found = db.search_lexical("Renal stone", limit=5)
                self.assertTrue(found)
                self.assertEqual(Path(found[0]["path"]), source.resolve())
            finally:
                db.close()

    def test_scholarly_locators_do_not_fake_pdf_page_numbers(self):
        base = dict(
            chunk_id=1,
            source_key="D1:P1:C0",
            title="paper",
            page=3,
            text="evidence",
            score=1.0,
        )
        self.assertEqual(
            answer_locator(Evidence(path="paper.nxml", **base)),
            "论文单元3",
        )
        self.assertEqual(
            answer_locator(Evidence(path="citation.ris", **base)),
            "文献记录3",
        )
        self.assertEqual(
            answer_locator(Evidence(path="paper.caj", **base)),
            "第3页",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
