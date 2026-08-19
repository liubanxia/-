from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "01_开发源码"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import pydicom  # noqa: F401
except ImportError:
    fake_pydicom = types.ModuleType("pydicom")

    def _unexpected_dcmread(*_args, **_kwargs):
        raise RuntimeError("simulated test should not require DICOM metadata")

    fake_pydicom.dcmread = _unexpected_dcmread
    sys.modules["pydicom"] = fake_pydicom

from core.pipeline import PhoenixPipeline


HEAD_MODELS = [
    "ich_2p5d_student",
    "ich_segmentation_student",
    "brain_infarct_2p5d_student",
    "brain_atrophy_quant_student",
]

ABDOMEN_MODELS = [
    "renal_stone_student",
    "sbo_2p5d_student",
    "appendicitis_2p5d_student",
]


class _FakeHub:
    def __init__(self, router_regions):
        self.status = {}
        self.errors = {}
        self.router_regions = list(router_regions)
        self.models = {
            "body_part_regression": object(),
            **{name: object() for name in HEAD_MODELS},
            **{name: object() for name in ABDOMEN_MODELS},
        }
        self.model_pool_policy = {}

    def load_selected(self, names):
        for name in names:
            self.status[name] = "loaded"

    def predict_selected(self, case, names):
        results = {}
        for name in names:
            if name == "body_part_regression":
                results[name] = {
                    "processed_images": 120,
                    "lesions": [],
                    "status": "success",
                    "inference_backend": "simulated_router",
                    "device": "cpu",
                    "active_body_regions": self.router_regions,
                    "body_part_examined_tag": " -> ".join(self.router_regions),
                }
            else:
                results[name] = {
                    "processed_images": 120,
                    "lesions": [],
                    "status": "success",
                    "inference_backend": f"simulated_{name}",
                    "device": "cpu",
                }
        return results


class CTSecondStageSimulationTest(unittest.TestCase):
    @staticmethod
    def _case(case_id="SIMULATED_CT"):
        series = types.SimpleNamespace(
            modality="CT",
            description="AXIAL SERIES",
            series_description="AXIAL SERIES",
            protocol_name="ROUTINE",
            study_description="",
            body_part="",
            files=[],
            series_uid="SIM-CT-001",
        )
        return types.SimpleNamespace(case_id=case_id, series=[series])

    def test_head_routes_to_lightweight_disease_and_segmentation_students(self):
        result = PhoenixPipeline(_FakeHub(["head"])).analyze(
            self._case("SIMULATED_HEAD_CT")
        )

        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", *HEAD_MODELS],
        )
        self.assertTrue(result["diagnostic_executed"])
        self.assertTrue(result["diagnostic_valid"])
        self.assertEqual(
            result["execution_summary"]["diagnostic_coverage"]["status"],
            "diagnostic_complete",
        )

    def test_abdomen_routes_to_lightweight_disease_students(self):
        result = PhoenixPipeline(_FakeHub(["abdomen"])).analyze(
            self._case("SIMULATED_ABDOMEN_CT")
        )

        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", *ABDOMEN_MODELS],
        )
        self.assertTrue(result["diagnostic_valid"])

    def test_chest_teacher_is_not_used_as_frontline_fallback(self):
        result = PhoenixPipeline(_FakeHub(["chest"])).analyze(
            self._case("SIMULATED_CHEST_CT")
        )

        self.assertEqual(result["selected_models"], ["body_part_regression"])
        route = result["execution_summary"]["routing"]
        self.assertEqual(route["second_stage_candidates"], [])
        self.assertFalse(result["diagnostic_executed"])
        self.assertFalse(result["diagnostic_valid"])
        self.assertEqual(
            result["execution_summary"]["diagnostic_coverage"]["regions_without_diagnostic_model"],
            ["chest"],
        )

    def test_multi_region_requires_coverage_for_every_routed_region(self):
        result = PhoenixPipeline(_FakeHub(["head", "chest"])).analyze(
            self._case("SIMULATED_HEAD_CHEST_CT")
        )

        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", *HEAD_MODELS],
        )
        coverage = result["execution_summary"]["diagnostic_coverage"]
        self.assertEqual(coverage["status"], "diagnostic_partial_coverage")
        self.assertEqual(coverage["regions_without_diagnostic_model"], ["chest"])
        self.assertFalse(result["diagnostic_valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
