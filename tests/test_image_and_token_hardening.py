from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from phoenix_knowledge.organizer import DeepOrganizer
from phoenix_knowledge.document_organizer import MultiDocumentOrganizer
from phoenix_knowledge.output_contracts import validate_export_bundle
from phoenix_knowledge.rich_export import MultiFormatExporter
from phoenix_knowledge.retrieval import Evidence


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


class _LLM:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt, max_new_tokens=1200, *, profile=None):
        self.calls.append((str(profile), int(max_new_tokens)))
        return prompt


class ImageAndTokenHardeningTests(unittest.TestCase):
    def test_rich_bundle_with_markdown_image_must_embed_docx_and_pdf_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "topic.md"
            assets = root / "topic_assets"
            assets.mkdir()
            (assets / "figure.png").write_bytes(_png_bytes())
            source.write_text(
                "# 图文专题\n\n![肺结节图](topic_assets/figure.png)\n",
                encoding="utf-8",
            )
            bundle = MultiFormatExporter(root / "out").export_path(source)
            report = validate_export_bundle(bundle)
            self.assertEqual(report["markdown"]["local_images"], 1)
            self.assertGreaterEqual(report["docx"]["images"], 1)
            self.assertGreaterEqual(report["pdf"]["images"], 1)

    def test_organizer_uses_fast_batches_and_one_quality_final_merge(self):
        organizer = DeepOrganizer.__new__(DeepOrganizer)
        organizer.llm = _LLM()
        partials = [f"证据笔记 {index} [S{index}]" for index in range(1, 9)]
        result = organizer._hierarchical_merge(
            "专题",
            "整理要求",
            partials,
            set(range(1, 9)),
        )
        self.assertIn("[S1]", result)
        profiles = [profile for profile, _budget in organizer.llm.calls]
        self.assertEqual(profiles, ["smart2", "smart2", "translation"])
        self.assertTrue(all(budget <= 1800 for _profile, budget in organizer.llm.calls))

    def test_real_multi_document_prompt_is_clipped_before_generation(self):
        evidence = [
            Evidence(
                chunk_id=index,
                source_key=f"D{index}",
                title=f"资料{index}",
                path=f"paper{index}.docx",
                page=index,
                text="medical evidence " * 1000,
                score=1.0,
            )
            for index in range(1, 13)
        ]
        prompt = MultiDocumentOrganizer._batch_prompt("整理肺结节", evidence)
        self.assertLessEqual(len(prompt), 12_500)
        self.assertIn("文档单元1", prompt)
        self.assertIn("token预算截断", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
