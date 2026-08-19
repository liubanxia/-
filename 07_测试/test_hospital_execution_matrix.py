from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "01_开发源码"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import pydicom  # noqa: F401
except ImportError:
    fake_pydicom = types.ModuleType("pydicom")

    def _dcmread(*_args, **_kwargs):
        return types.SimpleNamespace(
            BodyPartExamined="",
            StudyDescription="",
            SeriesDescription="",
            ProtocolName="",
        )

    fake_pydicom.dcmread = _dcmread
    sys.modules["pydicom"] = fake_pydicom

from core.hardware_profile import HardwareProfile
from core.model_hub import ModelHub
from core.pipeline import PhoenixPipeline


HOSPITAL_PROFILE = HardwareProfile(
    mode="hospital_light",
    ram_gb=8.0,
    gpu_name="NVIDIA Quadro K420",
    cuda_available=False,
    cuda_capability=None,
    inference_device="cpu",
    heavy_3d_allowed=False,
    reason="simulated i3-12100 / 8GB / Quadro K420 hospital workstation",
)


class _Model:
    def __init__(self, name, predictor):
        self.name = name
        self.predictor = predictor
        self.loaded = False
        self.predict_calls = 0

    def load(self):
        self.loaded = True

    def predict(self, case):
        self.predict_calls += 1
        return self.predictor(case)

    def unload(self):
        self.loaded = False


def _body_part(case):
    case_id = str(getattr(case, "case_id", "")).upper()
    if "CHEST" in case_id:
        regions = ["chest"]
    elif "HEAD" in case_id:
        regions = ["head"]
    elif "ABDOMEN" in case_id:
        regions = ["abdomen"]
    else:
        regions = []
    return {
        "status": "success",
        "processed_images": 96,
        "lesions": [],
        "device": "cpu",
        "inference_backend": "simulated_body_part_regression",
        "active_body_regions": regions,
        "body_part_examined_tag": " -> ".join(regions) if regions else "UNDETERMINED",
    }


def _simple_output(name, processed=1, *, backend=None):
    return lambda _case: {
        "status": "success",
        "processed_images": processed,
        "lesions": [],
        "device": "cpu",
        "inference_backend": backend or f"simulated_{name}",
    }


def _series(modality, *, body_part="", description="ROUTINE", series_uid="SIM-1"):
    return types.SimpleNamespace(
        modality=modality,
        body_part=body_part,
        description=description,
        series_description=description,
        study_description="",
        protocol_name="",
        files=[],
        series_uid=series_uid,
    )


def _case(case_id, series):
    return types.SimpleNamespace(
        case_id=case_id,
        series=list(series),
    )


class HospitalExecutionMatrixTests(unittest.TestCase):
    def _hub(self):
        with patch(
            "core.model_hub.detect_hardware_profile",
            return_value=HOSPITAL_PROFILE,
        ):
            hub = ModelHub()

        hub.register(_Model("body_part_regression", _body_part))
        hub.register(_Model(
            "blast_ct_head",
            _simple_output("blast_ct_head", processed=48, backend="simulated_blast_cpu"),
        ))
        hub.register(_Model(
            "monai_lung_nodule_ct",
            lambda _case: (_ for _ in ()).throw(
                AssertionError("MONAI must be hardware-deferred on hospital profile")
            ),
        ))
        hub.register(_Model(
            "torchxrayvision_chest",
            _simple_output("torchxrayvision_chest", backend="simulated_xrv_cpu"),
        ))
        hub.register(_Model(
            "fracture_rescbam",
            _simple_output("fracture_rescbam", backend="simulated_onnx_cpu"),
        ))
        hub.register(_Model(
            "fractureatlas_localization",
            _simple_output("fractureatlas_localization", backend="simulated_yolo_cpu"),
        ))
        hub.register(_Model(
            "fractureatlas_segmentation",
            _simple_output("fractureatlas_segmentation", backend="simulated_yolo_cpu"),
        ))

        # Representative registered-but-not-frontline components. The real full
        # hub contains more teacher/segmentation/experimental components; the
        # pipeline must make it explicit that they were not selected.
        hub.register(_Model("rad_dino", _simple_output("rad_dino")))
        hub.register(_Model("medsam2", _simple_output("medsam2")))
        hub.register(_Model("totalsegmentator", _simple_output("totalsegmentator")))
        return hub

    @staticmethod
    def _print_matrix(label, result):
        summary = result["execution_summary"]
        print(f"HOSPITAL_SIM_CASE={label}")
        print(f"HOSPITAL_SIM_SELECTED={result['selected_models']}")
        print(f"HOSPITAL_SIM_DIAGNOSTIC_EXECUTED={result['diagnostic_executed']}")
        print(f"HOSPITAL_SIM_DIAGNOSTIC_VALID={result['diagnostic_valid']}")
        print(f"HOSPITAL_SIM_ROUTING={summary.get('routing')}")
        for item in summary.get("models", []):
            # Keep CI logs ASCII-safe. Full localized error text remains in the
            # execution object and is asserted separately where required.
            print(
                "HOSPITAL_SIM_MODEL "
                f"name={item.get('model_name')} "
                f"status={item.get('status')} "
                f"executed={item.get('executed')} "
                f"device={item.get('device')} "
                f"backend={item.get('backend')} "
                f"has_error={bool(item.get('error'))}"
            )
        print(
            "HOSPITAL_SIM_NOT_SELECTED="
            + ",".join(summary.get("registered_models_not_selected", []))
        )

    def test_hospital_chest_ct_runs_router_but_defers_monai(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case("HOSPITAL_CHEST_CT", [_series("CT")])
        )
        self._print_matrix("CHEST_CT", result)

        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", "monai_lung_nodule_ct"],
        )
        executions = {
            item["model_name"]: item
            for item in result["execution_summary"]["models"]
        }
        self.assertTrue(executions["body_part_regression"]["executed"])
        self.assertEqual(
            executions["monai_lung_nodule_ct"]["status"],
            "hardware_deferred",
        )
        self.assertFalse(executions["monai_lung_nodule_ct"]["executed"])
        self.assertTrue(executions["monai_lung_nodule_ct"]["error"])
        self.assertFalse(result["diagnostic_executed"])
        self.assertFalse(result["diagnostic_valid"])
        self.assertIn("monai_lung_nodule_ct", result["incomplete_models"])

    def test_hospital_head_ct_runs_router_and_blast_cpu(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case("HOSPITAL_HEAD_CT", [_series("CT")])
        )
        self._print_matrix("HEAD_CT", result)

        self.assertEqual(
            result["selected_models"],
            ["body_part_regression", "blast_ct_head"],
        )
        self.assertTrue(result["diagnostic_executed"])
        self.assertTrue(result["diagnostic_valid"])

    def test_hospital_abdomen_ct_has_no_diagnostic_specialist_yet(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case("HOSPITAL_ABDOMEN_CT", [_series("CT")])
        )
        self._print_matrix("ABDOMEN_CT", result)

        self.assertEqual(result["selected_models"], ["body_part_regression"])
        self.assertFalse(result["diagnostic_executed"])
        self.assertFalse(result["diagnostic_valid"])

    def test_hospital_chest_dr_runs_screening_only(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case(
                "HOSPITAL_CHEST_DR",
                [_series("DX", body_part="CHEST")],
            )
        )
        self._print_matrix("CHEST_DR", result)

        self.assertEqual(result["selected_models"], ["torchxrayvision_chest"])
        self.assertEqual(
            result["execution_summary"]["screening_models_executed"],
            ["torchxrayvision_chest"],
        )
        self.assertFalse(result["diagnostic_executed"])
        self.assertFalse(result["diagnostic_valid"])

    def test_hospital_bone_dr_runs_fracture_diagnostic_and_helpers(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case(
                "HOSPITAL_BONE_DR",
                [_series("DX", body_part="HAND")],
            )
        )
        self._print_matrix("BONE_DR", result)

        self.assertEqual(
            result["selected_models"],
            [
                "fracture_rescbam",
                "fractureatlas_localization",
                "fractureatlas_segmentation",
            ],
        )
        self.assertTrue(result["diagnostic_executed"])
        self.assertTrue(result["diagnostic_valid"])
        self.assertEqual(
            result["execution_summary"]["diagnostic_models_executed"],
            ["fracture_rescbam"],
        )
        self.assertFalse(result["helper_failed_models"])

    def test_matrix_exposes_registered_but_not_selected_models(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case("HOSPITAL_ABDOMEN_CT", [_series("CT")])
        )
        not_selected = result["execution_summary"]["registered_models_not_selected"]
        self.assertIn("rad_dino", not_selected)
        self.assertIn("medsam2", not_selected)
        self.assertIn("totalsegmentator", not_selected)
        self.assertEqual(
            result["execution_summary"]["hardware_profile"]["mode"],
            "hospital_light",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
