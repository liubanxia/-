from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "01_开发源码"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.environment_paths import resolve_project_root


RUNTIME_FILES = [
    SRC_ROOT / "model_adapters" / "native_specialist_runtime.py",
    SRC_ROOT / "model_adapters" / "ct_segmentation_runtime.py",
    SRC_ROOT / "model_adapters" / "merlin_runtime.py",
    SRC_ROOT / "ai_models" / "dicom_inference_service.py",
    SRC_ROOT / "model_adapters" / "blast_ct.py",
    SRC_ROOT / "model_adapters" / "monai_lung_nodule_ct.py",
    SRC_ROOT / "core" / "yunpacs_live_controller.py",
    SRC_ROOT / "ui" / "phoenix_minimal_window.py",
]


class PortableRuntimePathTest(unittest.TestCase):

    def test_explicit_project_root_has_priority(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = resolve_project_root(temp_dir)
            self.assertEqual(path, Path(temp_dir).resolve())

    def test_runtime_code_does_not_hardcode_project_drive(self):
        forbidden = (
            "D:\\project_phoenix",
            "G:\\project_phoenix",
            "D:/project_phoenix",
            "G:/project_phoenix",
        )

        for path in RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(
                    token,
                    source,
                    msg=f"hard-coded project drive in {path}: {token}",
                )

    def test_legacy_ct_service_is_explicitly_non_diagnostic(self):
        source = (
            SRC_ROOT
            / "ai_models"
            / "dicom_inference_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "CT_ROUTING_COMPLETE_NO_DIAGNOSTIC",
            source,
        )
        self.assertIn(
            '"diagnostic_executed": False',
            source,
        )

    def test_live_controller_delegates_current_case_to_stable_cache_adapter(self):
        source = (
            SRC_ROOT
            / "core"
            / "yunpacs_live_controller.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"yunpacs",\n            "current"', source)
        self.assertNotIn("_today_case_dirs", source)
        self.assertNotIn("directory.stat().st_mtime_ns", source)

    def test_hospital_ui_never_reports_zero_when_diagnostic_not_executed(self):
        source = (
            SRC_ROOT
            / "ui"
            / "phoenix_minimal_window.py"
        ).read_text(encoding="utf-8")

        self.assertIn("DIAGNOSTIC_LESION_COUNT", source)
        self.assertIn('else "N/A"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
