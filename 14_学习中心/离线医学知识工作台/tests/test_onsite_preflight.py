from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "onsite_preflight.py"
SPEC = importlib.util.spec_from_file_location("phoenix_onsite_preflight", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class OnsitePreflightTests(unittest.TestCase):
    def test_common_environment_failure_blocks_everything(self):
        checks = [
            MODULE.Check("Python解释器", "FAIL", "wrong python"),
            MODULE.Check("核心依赖", "PASS", "ok"),
        ]
        state, causes = MODULE._classify(checks, {})
        self.assertEqual(state, "BLOCKED")
        self.assertTrue(any("基础运行环境" in item for item in causes))

    def test_missing_models_is_degraded_not_false_ready(self):
        checks = [
            MODULE.Check("Python解释器", "PASS", "ok"),
            MODULE.Check("核心依赖", "PASS", "ok"),
            MODULE.Check("项目写入", "PASS", "ok"),
            MODULE.Check("SSD空间", "PASS", "ok"),
            MODULE.Check("数据库", "PASS", "ok"),
            MODULE.Check("PDF输出引擎", "PASS", "ok"),
            MODULE.Check("GUI框架", "PASS", "ok"),
        ]
        capabilities = {
            "translation_backends": [],
            "smart1": False,
            "smart2": False,
            "semantic_ready": False,
        }
        state, causes = MODULE._classify(checks, capabilities)
        self.assertEqual(state, "DEGRADED")
        self.assertTrue(any("翻译后端" in item for item in causes))

    def test_all_core_capabilities_ready_is_ready(self):
        checks = [
            MODULE.Check("Python解释器", "PASS", "ok"),
            MODULE.Check("核心依赖", "PASS", "ok"),
            MODULE.Check("项目写入", "PASS", "ok"),
            MODULE.Check("SSD空间", "PASS", "ok"),
            MODULE.Check("数据库", "PASS", "ok"),
            MODULE.Check("PDF输出引擎", "PASS", "ok"),
            MODULE.Check("GUI框架", "PASS", "ok"),
            MODULE.Check("工作台能力", "PASS", "ok"),
        ]
        capabilities = {
            "translation_backends": ["local"],
            "smart1": True,
            "smart2": False,
            "semantic_ready": True,
        }
        state, causes = MODULE._classify(checks, capabilities)
        self.assertEqual(state, "READY")
        self.assertEqual(causes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
