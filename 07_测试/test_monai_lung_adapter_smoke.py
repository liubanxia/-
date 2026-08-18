from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPO_ROOT
    / "01_开发源码"
    / "model_adapters"
    / "monai_lung_nodule_ct.py"
)


class FakeModelAdapter:
    pass


class FakeTensor:
    def to(self, _device):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def tolist(self):
        return [0.0]

    def numel(self):
        return 1

    def __getitem__(self, _item):
        return self


class FakeNetwork:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self


class FakeDetector:
    def __init__(self):
        self.training = True
        self.eval_calls = 0
        self.network = None

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def __call__(self, images, use_inferer=False):
        if self.training:
            raise ValueError(
                "Please provide ground truth targets during training."
            )

        assert len(images) == 1
        assert use_inferer is False

        return [
            {
                "box": [[1, 2, 3, 4, 5, 6]],
                "label": [0],
                "label_scores": [0.91],
            }
        ]


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @staticmethod
    @contextmanager
    def inference_mode():
        yield


class MonaiLungAdapterSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = []

        model_adapter_mod = types.ModuleType("core.model_adapter")
        model_adapter_mod.ModelAdapter = FakeModelAdapter

        ct_nifti_mod = types.ModuleType("core.ct_nifti")
        ct_nifti_mod.series_to_nifti = lambda _series, output: Path(
            output
        ).write_bytes(b"fake")

        selector_mod = types.ModuleType("core.ct_series_selector")
        selector_mod.select_ct_series = (
            lambda case, anatomy, minimum_images=16: case
        )

        sys.modules["core"] = core_pkg
        sys.modules["core.model_adapter"] = model_adapter_mod
        sys.modules["core.ct_nifti"] = ct_nifti_mod
        sys.modules["core.ct_series_selector"] = selector_mod
        sys.modules["torch"] = FakeTorch()

        spec = importlib.util.spec_from_file_location(
            "phoenix_monai_lung_adapter",
            ADAPTER_PATH,
        )

        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        cls.module = module
        cls.Adapter = module.MonaiLungNoduleCTAdapter

    def test_source_no_longer_uses_bundle_cli(self):
        source = ADAPTER_PATH.read_text(encoding="utf-8")

        self.assertNotIn("subprocess.run", source)
        self.assertNotIn('"monai.bundle",\n            "run"', source)
        self.assertIn("self.detector.eval()", source)
        self.assertIn("self.detector.training = False", source)

    def test_direct_inference_forces_detector_eval(self):
        adapter = self.Adapter()
        adapter.device = "cpu"
        adapter.roi_size = (512, 512, 192)
        adapter.network = FakeNetwork()
        adapter.detector = FakeDetector()
        adapter.preprocessing = lambda _sample: {
            "image": FakeTensor()
        }
        adapter.postprocessing = lambda sample: sample

        self.module.series_to_nifti = (
            lambda _series, output: Path(output).write_bytes(b"fake")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = adapter._run_direct(
                series=object(),
                work_dir=Path(temp_dir),
            )

        self.assertFalse(adapter.detector.training)
        self.assertGreaterEqual(adapter.detector.eval_calls, 1)
        self.assertIs(adapter.detector.network, adapter.network)
        self.assertEqual(result["label"], [0])
        self.assertEqual(result["label_scores"], [0.91])
        self.assertEqual(
            result["box"],
            [[1, 2, 3, 4, 5, 6]],
        )

    def test_lesion_output_carries_lps_world_center_and_series_uid(self):
        adapter = self.Adapter()
        series = types.SimpleNamespace(series_uid="SERIES-1")

        lesions = adapter._to_lesions(
            {
                "box": [[10, 20, 30, 6, 8, 4]],
                "label": [0],
                "label_scores": [0.88],
            },
            series,
        )

        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0]["series_uid"], "SERIES-1")
        self.assertEqual(
            lesions[0]["world_point_lps"],
            [10.0, 20.0, 30.0],
        )
        self.assertEqual(
            lesions[0]["geometry_mode"],
            "cccwhd_lps",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
