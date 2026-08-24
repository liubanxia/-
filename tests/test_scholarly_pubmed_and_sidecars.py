from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.product_document_ingest import ProductDocumentIngestor
from phoenix_knowledge.scholarly_ingest import ScholarlyParser


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


class ScholarlyPubMedAndSidecarTests(unittest.TestCase):
    def test_pubmed_xml_supports_multiple_records_and_identifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "pubmed.xml"
            source.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>11111111</PMID>
   <Article>
    <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue><Title>Radiology</Title></Journal>
    <ArticleTitle>CT pulmonary nodule validation</ArticleTitle>
    <ELocationID EIdType="doi">10.1000/ct.111</ELocationID>
    <Abstract><AbstractText Label="RESULTS">AUC was 0.92.</AbstractText></Abstract>
    <AuthorList><Author><LastName>Zhang</LastName><ForeName>Wei</ForeName></Author></AuthorList>
   </Article>
   <KeywordList><Keyword>CT</Keyword><Keyword>nodule</Keyword></KeywordList>
  </MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="pubmed">11111111</ArticleId><ArticleId IdType="doi">10.1000/ct.111</ArticleId><ArticleId IdType="pmc">PMC1111111</ArticleId></ArticleIdList></PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>22222222</PMID>
   <Article>
    <Journal><JournalIssue><PubDate><MedlineDate>2024 Jan-Feb</MedlineDate></PubDate></JournalIssue><Title>European Radiology</Title></Journal>
    <ArticleTitle>MRI liver lesion study</ArticleTitle>
    <Abstract><AbstractText>External validation included 430 patients.</AbstractText></Abstract>
   </Article>
  </MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>""",
                encoding="utf-8",
            )

            parsed = ScholarlyParser(root / "runtime").parse(source)
            self.assertEqual(len(parsed.records), 2)
            self.assertEqual(len(parsed.units), 2)
            self.assertEqual(parsed.records[0].title, "CT pulmonary nodule validation")
            self.assertEqual(parsed.records[0].pmid, "11111111")
            self.assertEqual(parsed.records[0].doi, "10.1000/ct.111")
            self.assertEqual(parsed.records[0].pmcid, "PMC1111111")
            self.assertIn("AUC was 0.92", parsed.records[0].abstract)
            self.assertEqual(parsed.records[1].year, "2024")
            self.assertIn("430 patients", parsed.units[1].text)

    def test_jats_sidecar_figure_survives_copy_into_library(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external = root / "downloads"
            external.mkdir()
            (external / "figure1.png").write_bytes(_png_bytes())
            source = external / "article.nxml"
            source.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
 <front><article-meta>
  <article-id pub-id-type="doi">10.9999/sidecar.1</article-id>
  <title-group><article-title>Sidecar Figure Study</article-title></title-group>
 </article-meta></front>
 <body><sec><title>Results</title><p>Figure demonstrates the lesion.</p>
  <fig><graphic xlink:href="figure1.png"/></fig>
 </sec></body>
</article>""",
                encoding="utf-8",
            )

            paths = _paths(root)
            db = KnowledgeDB(paths.database)
            try:
                ingestor = ProductDocumentIngestor(db, paths)
                result = ingestor.ingest(
                    source,
                    copy_into_library=True,
                    extract_images=True,
                )
                self.assertTrue(
                    os.path.samefile(
                        result.copied_to_library.parent,
                        paths.source_root,
                    )
                )
                self.assertGreater(result.image_count, 0)
                doc_root = ingestor.assets.document_root(result.copied_to_library)
                self.assertTrue(any(doc_root.rglob("*.png")))
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
