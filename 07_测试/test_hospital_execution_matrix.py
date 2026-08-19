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
from core.model_pool_policy import ARCHIVED_FROM_FRONTLINE, attach_model_pool_policy
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

    def load(self):
        self.loaded = True

    def predict(self, case):
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


def _simple_output(name, processed=1, *, status="success"):
    return lambda _case: {
        "status": status,
        "processed_images": processed if status == "success" else 0,
        "lesions": [],
        "device": "cpu",
        "inference_backend": f"simulated_{name}",
        **({"error": "simulated failure"} if status != "success" else {}),
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
    return types.SimpleNamespace(case_id=case_id, series=list(series))


class HospitalExecutionMatrixTests(unittest.TestCase):
    def _hub(self, *, fail_fracture_segmentation=False):
        with patch(
            "core.model_hub.detect_hardware_profile",
            return_value=HOSPITAL_PROFILE,
        ):
            hub = ModelHub()

        hub.register(_Model("body_part_regression", _body_part))
        hub.register(_Model(
            "fracture_rescbam",
            _simple_output("fracture_rescbam"),
        ))
        hub.register(_Model(
            "fractureatlas_localization",
            _simple_output("fractureatlas_localization"),
        ))
        hub.register(_Model(
            "fractureatlas_segmentation",
            _simple_output(
                "fractureatlas_segmentation",
                status="error" if fail_fracture_segmentation else "success",
            ),
        ))
        return attach_model_pool_policy(hub)

    def test_hospital_head_ct_exposes_student_gap_and_never_falls_back_to_blast(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case("HOSPITAL_HEAD_CT", [_series("CT")])
        )

        self.assertEqual(result["selected_models"], ["body_part_regression"])
        route = result["execution_summary"]["routing"]
        self.assertIn("ich_2p5d_student", route["unavailable_second_stage_models"])
        self.assertNotIn("blast_ct_head", hub.models)
        self.assertFalse(result["diagnostic_valid"])
        self.assertEqual(
            result["execution_summary"]["diagnostic_coverage"]["regions_without_diagnostic_model"],
            ["head"],
        )

    def test_hospital_chest_ct_keeps_teacher_out_of_frontline(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case("HOSPITAL_CHEST_CT", [_series("CT")])
        )

        self.assertEqual(result["selected_models"], ["body_part_regression"])
        self.assertNotIn("monai_lung_nodule_ct", hub.models)
        self.assertFalse(result["diagnostic_valid"])
        self.assertEqual(
            result["execution_summary"]["diagnostic_coverage"]["regions_without_diagnostic_model"],
            ["chest"],
        )

    def test_hospital_abdomen_ct_exposes_planned_students_as_unavailable(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case("HOSPITAL_ABDOMEN_CT", [_series("CT")])
        )

        route = result["execution_summary"]["routing"]
        self.assertEqual(
            route["second_stage_candidates"],
            ["renal_stone_student", "sbo_2p5d_student", "appendicitis_2p5d_student"],
        )
        self.assertEqual(result["selected_models"], ["body_part_regression"])
        self.assertFalse(result["diagnostic_valid"])

    def test_hospital_chest_dr_points_to_nano_detector_without_screening_fallback(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case(
                "HOSPITAL_CHEST_DR",
                [_series("DX", body_part="CHEST")],
            )
        )

        route = result["execution_summary"]["routing"]
        self.assertEqual(route["unavailable_initial_models"], ["chest_dr_nano_detector"])
        self.assertEqual(result["selected_models"], [])
        self.assertNotIn("torchxrayvision_chest", hub.models)
        self.assertFalse(result["diagnostic_valid"])

    def test_hospital_bone_dr_requires_detection_localization_and_segmentation(self):
        hub = self._hub()
        result = PhoenixPipeline(hub).analyze(
            _case(
                "HOSPITAL_BONE_DR",
                [_series("DX", body_part="HAND")],
            )
        )

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

    def test_failed_fracture_segmentation_invalidates_clinical_chain(self):
        hub = self._hub(fail_fracture_segmentation=True)
        result = PhoenixPipeline(hub).analyze(
            _case(
                "HOSPITAL_BONE_DR_FAIL_SEG",
                [_series("DX", body_part="HAND")],
            )
        )

        self.assertTrue(result["diagnostic_executed"])
        self.assertFalse(result["diagnostic_valid"])
        self.assertEqual(
            result["execution_summary"]["diagnostic_coverage"]["status"],
            "diagnostic_spatial_chain_incomplete",
        )

    def test_archived_and_teacher_models_are_absent_from_hospital_frontline_pool(self):
        hub = self._hub()
        for name in ARCHIVED_FROM_FRONTLINE:
            self.assertNotIn(name, hub.models)

        self.assertEqual(
            set(hub.models),
            {
                "body_part_regression",
                "fracture_rescbam",
                "fractureatlas_localization",
                "fractureatlas_segmentation",
            },
        )
        self.assertEqual(hub.model_pool_policy["hardware_mode"], "hospital_light")


if __name__ == "__main__":
    unittest.main(verbosity=2)
