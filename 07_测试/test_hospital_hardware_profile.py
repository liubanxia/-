from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "01_开发源码"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from core.hardware_profile import HardwareProfile
from core.model_hub import ModelHub


class _HeavyModel:
    name = "monai_lung_nodule_ct"

    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True

    def predict(self, case):
        raise AssertionError("hospital hardware guard should prevent prediction")

    def unload(self):
        pass


class HospitalHardwareGuardTests(unittest.TestCase):
    def test_k420_low_memory_profile_defers_heavy_3d_model(self):
        profile = HardwareProfile(
            mode="hospital_light",
            ram_gb=8.0,
            gpu_name="NVIDIA Quadro K420",
            cuda_available=False,
            cuda_capability=None,
            inference_device="cpu",
            heavy_3d_allowed=False,
            reason="test",
        )

        with patch("core.model_hub.detect_hardware_profile", return_value=profile):
            hub = ModelHub()

        model = _HeavyModel()
        hub.register(model)
        hub.load_selected([model.name])

        self.assertFalse(model.loaded)
        self.assertEqual(hub.status[model.name], "hardware_deferred")
        self.assertIn("PHOENIX_ALLOW_HEAVY_CPU=1", hub.errors[model.name])

        result = hub.predict_selected(object(), [model.name])[model.name]
        self.assertEqual(result["status"], "hardware_deferred")
        self.assertIn("硬件保护", result["error"])

    def test_explicit_heavy_cpu_override_allows_load(self):
        profile = HardwareProfile(
            mode="hospital_light",
            ram_gb=8.0,
            gpu_name="NVIDIA Quadro K420",
            cuda_available=False,
            cuda_capability=None,
            inference_device="cpu",
            heavy_3d_allowed=True,
            reason="manual override",
        )

        with patch("core.model_hub.detect_hardware_profile", return_value=profile):
            hub = ModelHub()

        model = _HeavyModel()
        hub.register(model)
        hub.load_selected([model.name])

        self.assertTrue(model.loaded)
        self.assertEqual(hub.status[model.name], "loaded")


if __name__ == "__main__":
    unittest.main()
