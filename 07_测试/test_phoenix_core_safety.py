from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "01_开发源码"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.contracts import AnalysisResult
from core.ct_series_selector import select_ct_series
from core.dicom_geometry import DicomPlaneGeometry, world_lps_to_pixel
from core.execution_status import execution_from_raw
from core.report_generator import generate_report
from core.result_fusion import fuse_results


class PhoenixCoreSafetyTest(unittest.TestCase):

    def test_monai_box_is_preserved_as_lps_world_center(self):
        result = fuse_results(
            {
                "monai_lung_nodule_ct": {
                    "processed_images": 100,
                    "lesions": [
                        {
                            "finding": "肺结节候选灶",
                            "score": 0.91,
                            "label": 0,
                            "geometry": {
                                "box_3d": [10, 20, 30, 6, 8, 4],
                            },
                        }
                    ],
                }
            }
        )

        self.assertEqual(len(result.lesions), 1)
        lesion = result.lesions[0]
        self.assertEqual(lesion.label, "肺结节候选灶")
        self.assertAlmostEqual(lesion.confidence, 0.91)
        self.assertEqual(lesion.world_point_lps, (10.0, 20.0, 30.0))
        self.assertEqual(lesion.geometry_mode, "cccwhd_lps")

    def test_lps_to_pixel_mapping_uses_dicom_spacing_order(self):
        geometry = DicomPlaneGeometry(
            path=Path("test.dcm"),
            image_position=(0.0, 0.0, 0.0),
            row_direction=(1.0, 0.0, 0.0),
            column_direction=(0.0, 1.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            row_spacing=2.0,
            column_spacing=1.0,
            rows=512,
            columns=512,
            instance_number=1,
        )

        x, y, offset = world_lps_to_pixel(
            (10.0, 6.0, 0.0),
            geometry,
        )

        self.assertAlmostEqual(x, 10.0)
        self.assertAlmostEqual(y, 3.0)
        self.assertAlmostEqual(offset, 0.0)

    def test_chest_specialist_does_not_pick_longer_abdominal_series(self):
        abdomen = types.SimpleNamespace(
            modality="CT",
            series_description="ABDOMEN PELVIS",
            protocol_name="ABDOMEN",
            files=[Path(f"a_{i}.dcm") for i in range(500)],
        )
        chest = types.SimpleNamespace(
            modality="CT",
            series_description="CHEST LUNG",
            protocol_name="THORAX",
            files=[Path(f"c_{i}.dcm") for i in range(120)],
        )
        case = types.SimpleNamespace(series=[abdomen, chest])

        selected = select_ct_series(case, "chest")
        self.assertIs(selected, chest)

    def test_head_specialist_does_not_pick_longer_chest_series(self):
        chest = types.SimpleNamespace(
            modality="CT",
            series_description="CHEST THORAX",
            protocol_name="LUNG",
            files=[Path(f"c_{i}.dcm") for i in range(500)],
        )
        head = types.SimpleNamespace(
            modality="CT",
            series_description="HEAD BRAIN",
            protocol_name="HEAD",
            files=[Path(f"h_{i}.dcm") for i in range(80)],
        )
        case = types.SimpleNamespace(series=[chest, head])

        selected = select_ct_series(case, "head")
        self.assertIs(selected, head)

    def test_specialist_refuses_unmatched_body_part(self):
        abdomen = types.SimpleNamespace(
            modality="CT",
            series_description="ABDOMEN PELVIS",
            protocol_name="ABDOMEN",
            files=[Path(f"a_{i}.dcm") for i in range(200)],
        )
        case = types.SimpleNamespace(series=[abdomen])

        with self.assertRaises(RuntimeError):
            select_ct_series(case, "chest")
        with self.assertRaises(RuntimeError):
            select_ct_series(case, "head")

    def test_localizer_is_never_selected_as_specialist_series(self):
        scout = types.SimpleNamespace(
            modality="CT",
            series_description="CHEST SCOUT LOCALIZER",
            protocol_name="CHEST",
            files=[Path(f"s_{i}.dcm") for i in range(40)],
        )
        diagnostic = types.SimpleNamespace(
            modality="CT",
            series_description="CHEST AXIAL LUNG",
            protocol_name="THORAX",
            files=[Path(f"d_{i}.dcm") for i in range(100)],
        )
        case = types.SimpleNamespace(series=[scout, diagnostic])

        self.assertIs(select_ct_series(case, "chest"), diagnostic)

    def test_failed_model_is_never_treated_as_executed_negative(self):
        execution = execution_from_raw(
            "monai_lung_nodule_ct",
            {
                "error": "boom",
                "status": "failed",
            },
            load_status="loaded",
        )

        self.assertFalse(execution.executed)
        self.assertIsNone(execution.lesion_count)
        self.assertFalse(execution.valid_negative)

    def test_successful_zero_lesion_model_is_explicit_valid_negative_state(self):
        execution = execution_from_raw(
            "monai_lung_nodule_ct",
            {
                "processed_images": 100,
                "lesions": [],
            },
            load_status="loaded",
        )

        self.assertTrue(execution.executed)
        self.assertEqual(execution.lesion_count, 0)
        self.assertTrue(execution.valid_negative)

    def test_report_blocks_router_only_ct_from_looking_negative(self):
        result = AnalysisResult()
        result.execution_summary = {
            "diagnostic_models_selected": [],
            "screening_models_executed": [],
        }

        generate_report(result, incomplete_models=[])

        self.assertIn("尚无适用的疾病诊断模型", result.report_draft)
        self.assertNotIn("未见明显异常", result.report_draft)

    def test_report_zero_lesion_diagnostic_is_not_called_negative_diagnosis(self):
        result = AnalysisResult()
        result.execution_summary = {
            "diagnostic_models_selected": ["monai_lung_nodule_ct"],
            "screening_models_executed": [],
        }
        result.diagnostic_executed = True
        result.diagnostic_valid = True

        generate_report(result, incomplete_models=[])

        self.assertIn("未输出候选病灶", result.report_draft)
        self.assertIn("不等同于影像学阴性诊断", result.report_draft)


if __name__ == "__main__":
    unittest.main(verbosity=2)
