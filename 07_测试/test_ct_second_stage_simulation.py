from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "01_开发源码"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# GitHub's lightweight smoke environment intentionally does not install
# pydicom. These tests prove that BodyPartRegression can drive routing even
# when DICOM descriptive tags are absent.
try:
    import pydicom  # noqa: F401
except ImportError:
    fake_pydicom = types.ModuleType("pydicom")

    def _unexpected_dcmread(*_args, **_kwargs):
        raise RuntimeError("simulated test should not require DICOM metadata")

    fake_pydicom.dcmread = _unexpected_dcmread
    sys.modules["pydicom"] = fake_pydicom

from core.pipeline import PhoenixPipeline


class _FakeHub:
    def __init__(self, router_regions):
        self.status = {}
        self.errors = {}
        self.router_regions = list(router_regions)
        self.models = {
            "body_part_regression": object(),
            "blast_ct_head": object(),
            "monai_lung_nodule_ct": object(),
        }

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
            elif name == "monai_lung_nodule_ct":
                results[name] = {
                    "processed_images": 120,
                    "lesions": [],
                    "status": "success",
                    "inference_backend": "simulated_monai_retinanet",
                    "device": "cuda:0",
                }
            elif name == "blast_ct_head":
                results[name] = {
                    "processed_images": 48,
                    "lesions": [],
                    "status": "success",
                    "inference_backend": "simulated_blast_ct",
                    "device": "cpu",
                }
            else:
                raise AssertionError(f"unexpected simulated model: {name}")
        return results


class CTSecondStageSimulationTest(unittest.TestCase):
    @staticmethod
    def _case(case_id="SIMULATED_CT"):
        # Deliberately use non-informative metadata. Old Phoenix stopped after
        # BodyPartRegression here because specialist selection happened before
        # the router result existed.
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

    def test_bpr_chest_output_routes_and_executes_lung_specialist(self):
        result = PhoenixPipeline(_FakeHub(["chest"])).analyze(
            self._case("SIMULATED_CHEST_CT")
        )

        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", "monai_lung_nodule_ct"],
        )
        self.assertTrue(result["diagnostic_executed"])
        self.assertTrue(result["diagnostic_valid"])

        route = result["execution_summary"]["routing"]
        self.assertEqual(route["mode"], "ct_two_stage")
        self.assertEqual(route["ct_decision"]["router_regions"], ["chest"])
        self.assertTrue(route["ct_decision"]["chest"])
        self.assertFalse(route["ct_decision"]["metadata_chest"])

        executions = {
            item["model_name"]: item
            for item in result["execution_summary"]["models"]
        }
        monai = executions["monai_lung_nodule_ct"]
        self.assertTrue(monai["executed"])
        self.assertEqual(monai["status"], "success")
        self.assertEqual(monai["processed_images"], 120)
        self.assertEqual(monai["lesion_count"], 0)
        self.assertEqual(monai["device"], "cuda:0")

        print("SIMULATED_CASE=SIMULATED_CHEST_CT")
        print(f"SIMULATED_SELECTED_MODELS={result['selected_models']}")
        print(f"SIMULATED_ROUTE={route}")
        print(f"SIMULATED_DIAGNOSTIC_EXECUTED={result['diagnostic_executed']}")
        print(f"SIMULATED_DIAGNOSTIC_VALID={result['diagnostic_valid']}")

    def test_bpr_head_output_routes_to_blast_ct(self):
        result = PhoenixPipeline(_FakeHub(["head"])).analyze(
            self._case("SIMULATED_HEAD_CT")
        )
        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", "blast_ct_head"],
        )
        self.assertTrue(result["diagnostic_executed"])
        self.assertTrue(result["diagnostic_valid"])

    def test_bpr_multi_region_output_can_route_head_and_chest(self):
        result = PhoenixPipeline(_FakeHub(["chest", "head"])).analyze(
            self._case("SIMULATED_HEAD_CHEST_CT")
        )
        self.assertEqual(
            result["selected_models"],
            [
                "body_part_regression",
                "blast_ct_head",
                "monai_lung_nodule_ct",
            ],
        )
        self.assertTrue(result["diagnostic_valid"])

    def test_abdomen_ct_remains_router_only_until_abdominal_specialist_exists(self):
        result = PhoenixPipeline(_FakeHub(["abdomen"])).analyze(
            self._case("SIMULATED_ABDOMEN_CT")
        )

        self.assertEqual(result["selected_models"], ["body_part_regression"])
        self.assertFalse(result["diagnostic_executed"])
        self.assertFalse(result["diagnostic_valid"])
        self.assertEqual(
            result["execution_summary"]["routing"]["ct_decision"]["router_regions"],
            ["abdomen"],
        )

        print("SIMULATED_CASE=SIMULATED_ABDOMEN_CT")
        print(f"SIMULATED_SELECTED_MODELS={result['selected_models']}")
        print(f"SIMULATED_DIAGNOSTIC_EXECUTED={result['diagnostic_executed']}")
        print(f"SIMULATED_DIAGNOSTIC_VALID={result['diagnostic_valid']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
