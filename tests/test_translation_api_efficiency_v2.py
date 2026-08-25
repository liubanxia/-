from __future__ import annotations

import os
import unittest

from phoenix_knowledge import office_translation, translator
from phoenix_knowledge.office_translation import OfficeSegment
from phoenix_knowledge.translation_api_efficiency_v2 import (
    api_efficiency_stats,
    install,
)
from phoenix_knowledge.translation_models import QwenMedicalTranslationBackend


class _Compute:
    def requested_mode(self):
        return "remote"

    def provider_id(self):
        return "deepseek"

    def remote_model(self, _profile=None):
        return "deepseek-v4-pro"


class _LLM:
    def __init__(self):
        self.compute = _Compute()
        self.calls = 0
        self.prompts: list[str] = []
        self.batch_response = (
            '[{"id":"S0001","translation":"第一条医学译文"},'
            '{"id":"S0002","translation":"第二条医学译文"}]'
        )

    def backend(self, _profile=None):
        return "remote_server"

    def available(self, _profile=None):
        return True

    def generate(self, prompt, **_kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "输入JSON" in str(prompt):
            return self.batch_response
        return "规范医学译文"


class TranslationAPIEfficiencyV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install()

    def setUp(self):
        self.saved = dict(os.environ)
        os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"] = "1"
        os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "1"
        os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "remote"
        os.environ.pop("PHOENIX_TRANSLATION_CHUNK_CHARS", None)
        os.environ.pop("PHOENIX_OFFICE_API_BATCH_CHARS", None)
        os.environ.pop("PHOENIX_OFFICE_API_BATCH_SEGMENTS", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved)

    def test_exact_remote_source_is_called_once_per_session(self):
        llm = _LLM()
        backend = QwenMedicalTranslationBackend(llm)
        source = (
            "No evidence of acute intracranial hemorrhage or mass effect is identified "
            "on the current examination."
        )
        first = backend.translate(source, "中文", smart_level="smart2")
        second = backend.translate(source, "中文", smart_level="smart2")
        self.assertEqual(first, second)
        self.assertEqual(llm.calls, 1)
        stats = api_efficiency_stats(backend)
        self.assertEqual(stats["remote_calls"], 1)
        self.assertGreaterEqual(stats["cache_hits"], 1)

        # The compact prompt must retain the safety-critical invariants while
        # removing the old long repeated instruction block.
        prompt = llm.prompts[0]
        for marker in ("数字", "单位", "侧别", "否定", "医学缩写", "禁止总结"):
            self.assertIn(marker, prompt)
        self.assertLess(len(prompt) - len(source), 520)

    def test_exact_remote_batch_is_cached(self):
        llm = _LLM()
        backend = QwenMedicalTranslationBackend(llm)
        sources = [
            "No acute intracranial hemorrhage is identified.",
            "There is no significant midline shift or mass effect.",
        ]
        first = backend.translate_segments(sources, "中文")
        second = backend.translate_segments(sources, "中文")
        self.assertEqual(first, second)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(set(first), {"S0001", "S0002"})

    def test_second_generic_retry_is_avoided_but_safety_retry_is_kept(self):
        llm = _LLM()
        backend = QwenMedicalTranslationBackend(llm)
        source = "A medically meaningful source sentence for correction."

        first = backend.retry_translation(
            source,
            "初稿",
            ("一般语言质量问题",),
            "中文",
        )
        second = backend.retry_translation(
            source,
            "另一初稿",
            ("一般语言质量问题",),
            "中文",
        )
        self.assertEqual(first, second)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(api_efficiency_stats(backend)["retry_calls_avoided"], 1)

        backend.retry_translation(
            source,
            "安全修订初稿",
            ("数字/单位/正负号未完整保留",),
            "中文",
        )
        self.assertEqual(llm.calls, 2)

    def test_remote_office_batches_are_larger_without_affecting_local_default(self):
        segments = [
            OfficeSegment(f"S{i}", "word/document.xml", i, "A" * 1500)
            for i in range(3)
        ]
        remote_batches = office_translation._segment_batches(segments)
        self.assertEqual(len(remote_batches), 1)

        os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"] = "0"
        local_batches = office_translation._segment_batches(segments)
        self.assertEqual(len(local_batches), 3)

    def test_remote_pdf_chunk_budget_reduces_prompt_repetition(self):
        self.assertEqual(translator._translation_chunk_chars(), 6800)
        os.environ["PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"] = "0"
        self.assertEqual(translator._translation_chunk_chars(), 4800)


if __name__ == "__main__":
    unittest.main()
