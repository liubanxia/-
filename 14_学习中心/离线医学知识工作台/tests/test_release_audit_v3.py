from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from phoenix_knowledge.release_gui_truth import install as install_gui_truth


class _Emitter:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class _Answer:
    def __init__(self, text: str, mode: str):
        self.text = text
        self.mode = mode


class _Workbench:
    def __init__(self, final_mode: str):
        self.calls = 0
        self.final_mode = final_mode

    def ask(self, query, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _Answer("即时证据 [S1]", "evidence_only")
        return _Answer("最终内容 [S1]", self.final_mode)


class _AskWorker:
    def __init__(self, final_mode: str):
        self.workbench = _Workbench(final_mode)
        self.query = "测试问题"
        self.completed = _Emitter()
        self.failed = _Emitter()


class _Module:
    AskWorker = _AskWorker


class ReleaseAuditV3Tests(unittest.TestCase):
    def setUp(self):
        import phoenix_knowledge.release_gui_truth as truth

        truth._INSTALLED = False
        install_gui_truth(_Module)

    def test_smart_mode_fallback_is_not_labeled_as_successful_generation(self):
        worker = _AskWorker("evidence_only")
        with patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_DEEP_QA": "1",
                "PHOENIX_KNOWLEDGE_LLM_PROFILE": "fast",
            },
            clear=False,
        ):
            worker.run()
        self.assertFalse(worker.failed.values)
        self.assertEqual(len(worker.completed.values), 2)
        final = worker.completed.values[-1]
        self.assertIn("智能1未实际生成", final)
        self.assertIn("已回退资料证据", final)

    def test_grounded_generation_is_labeled_as_actual_smart_result(self):
        worker = _AskWorker("grounded_generation")
        with patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_DEEP_QA": "1",
                "PHOENIX_KNOWLEDGE_LLM_PROFILE": "deep",
            },
            clear=False,
        ):
            worker.run()
        self.assertFalse(worker.failed.values)
        final = worker.completed.values[-1]
        self.assertIn("完成 | 智能2", final)
        self.assertNotIn("未实际生成", final)

    def test_grounding_blocked_is_explicitly_reported(self):
        worker = _AskWorker("grounding_blocked")
        with patch.dict(
            os.environ,
            {
                "PHOENIX_KNOWLEDGE_DEEP_QA": "1",
                "PHOENIX_KNOWLEDGE_LLM_PROFILE": "fast",
            },
            clear=False,
        ):
            worker.run()
        final = worker.completed.values[-1]
        self.assertIn("引用安全门拦截", final)
        self.assertIn("回退资料证据", final)


if __name__ == "__main__":
    unittest.main(verbosity=2)
