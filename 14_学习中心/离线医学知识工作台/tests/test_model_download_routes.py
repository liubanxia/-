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
            target = Path(td) / "opus-mt-en-zh"
            target.mkdir()
            (target / "config.json").write_text("{}", encoding="utf-8")
            ok, missing = model_download.validate_download("translation_fast", target)
            self.assertFalse(ok)
            self.assertTrue(missing)

    def test_validation_accepts_minimal_marian_layout(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "opus-mt-en-zh"
            target.mkdir()
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "source.spm").write_bytes(b"source")
            (target / "target.spm").write_bytes(b"target")
            (target / "pytorch_model.bin").write_bytes(b"weights")
            ok, missing = model_download.validate_download("translation_fast", target)
            self.assertTrue(ok, missing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
