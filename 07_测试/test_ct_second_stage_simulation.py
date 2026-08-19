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
# pydicom. The simulated chest case is identified from SeriesDescription, so
# no DICOM file read is required; provide only the import surface used by the
# router modules when pydicom is absent.
try:
    import pydicom  # noqa: F401
except ImportError:
    fake_pydicom = types.ModuleType("pydicom")

    def _unexpected_dcmread(*_args, **_kwargs):
        raise RuntimeError("simulated test should not read a DICOM file")

    fake_pydicom.dcmread = _unexpected_dcmread
    sys.modules["pydicom"] = fake_pydicom

from core.pipeline import PhoenixPipeline


class _FakeHub:
    def __init__(self):
        self.status = {}
        self.errors = {}

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
                }
            elif name == "monai_lung_nodule_ct":
                results[name] = {
                    "processed_images": 120,
                    "lesions": [],
                    "status": "success",
                    "inference_backend": "simulated_monai_retinanet",
                    "device": "cuda:0",
                }
            else:
                raise AssertionError(f"unexpected simulated model: {name}")
        return results


class CTSecondStageSimulationTest(unittest.TestCase):
    def test_chest_ct_routes_and_executes_second_stage_diagnostic_model(self):
        chest = types.SimpleNamespace(
            modality="CT",
            series_description="CHEST AXIAL LUNG",
            protocol_name="THORAX",
            files=[],
            series_uid="SIM-CHEST-001",
        )
        case = types.SimpleNamespace(
            case_id="SIMULATED_CHEST_CT",
            series=[chest],
        )

        result = PhoenixPipeline(_FakeHub()).analyze(case)

        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", "monai_lung_nodule_ct"],
        )
        self.assertTrue(result["diagnostic_executed"])
        self.assertTrue(result["diagnostic_valid"])

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
        print(f"SIMULATED_DIAGNOSTIC_EXECUTED={result['diagnostic_executed']}")
        print(f"SIMULATED_DIAGNOSTIC_VALID={result['diagnostic_valid']}")
        print(
            "SIMULATED_MONAI="
            f"status={monai['status']} "
            f"executed={monai['executed']} "
            f"processed={monai['processed_images']} "
            f"lesions={monai['lesion_count']} "
            f"device={monai['device']}"
        )

    def test_non_chest_ct_does_not_claim_diagnostic_execution(self):
        abdomen = types.SimpleNamespace(
            modality="CT",
            series_description="ABDOMEN PELVIS",
            protocol_name="ABDOMEN",
            files=[],
            series_uid="SIM-ABD-001",
        )
        case = types.SimpleNamespace(
            case_id="SIMULATED_ABDOMEN_CT",
            series=[abdomen],
        )

        result = PhoenixPipeline(_FakeHub()).analyze(case)

        self.assertEqual(result["selected_models"], ["body_part_regression"])
        self.assertFalse(result["diagnostic_executed"])
        self.assertFalse(result["diagnostic_valid"])

        print("SIMULATED_CASE=SIMULATED_ABDOMEN_CT")
        print(f"SIMULATED_SELECTED_MODELS={result['selected_models']}")
        print(f"SIMULATED_DIAGNOSTIC_EXECUTED={result['diagnostic_executed']}")
        print(f"SIMULATED_DIAGNOSTIC_VALID={result['diagnostic_valid']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
