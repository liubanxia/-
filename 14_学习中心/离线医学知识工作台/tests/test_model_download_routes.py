from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phoenix_model_download", ROOT / "model_download.py")
model_download = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(model_download)


class ModelDownloadRouteTest(unittest.TestCase):
    def test_auto_route_order_prefers_modelscope_then_mirror_then_official(self):
        routes = model_download.build_routes(
            "auto", mirrors=["https://mirror-a.example", "https://mirror-b.example"]
        )
        self.assertEqual(
            routes,
            [
                ("modelscope", None),
                ("hf-mirror", "https://mirror-a.example"),
                ("hf-mirror", "https://mirror-b.example"),
                ("huggingface", "https://huggingface.co"),
            ],
        )

    def test_validation_rejects_incomplete_model(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "Qwen3.5-4B"
            target.mkdir()
            (target / "config.json").write_text("{}", encoding="utf-8")
            ok, missing = model_download.validate_download("generator", target)
            self.assertFalse(ok)
            self.assertTrue(missing)

    def test_validation_accepts_minimal_smart2_layout(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "Qwen3.5-4B"
            target.mkdir()
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors").write_bytes(b"weights")
            ok, missing = model_download.validate_download("generator", target)
            self.assertTrue(ok, missing)

    def test_formal_translation_groups_exclude_legacy_smart1_models(self):
        self.assertEqual(model_download.GROUPS["translation"], ["generator"])
        self.assertNotIn("translation_fast", model_download.MODELS)
        self.assertNotIn("translation_backup", model_download.MODELS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
