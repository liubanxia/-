from __future__ import annotations

import unittest

from phoenix_knowledge import translation_portable_local_runtime as portable


class PortableLocalTranslationRuntimeTest(unittest.TestCase):
    def test_runtime_reuses_generic_hardware_selector(self):
        code = open(portable.__file__, "r", encoding="utf-8").read()
        self.assertIn("choose_device", code)
        self.assertIn("_cuda_probe", code)
        self.assertNotIn("RTX 5060", code)
        self.assertNotIn("NVIDIA GeForce RTX 5060", code)

    def test_model1_backends_are_patched(self):
        code = open(portable.__file__, "r", encoding="utf-8").read()
        self.assertIn("MarianEnZhBackend._load = marian_load", code)
        self.assertIn("NLLBEnZhBackend._load = nllb_load", code)

    def test_model2_backend_is_patched(self):
        code = open(portable.__file__, "r", encoding="utf-8").read()
        self.assertIn("HYMTMedicalTranslationBackend._load = _load_hymt", code)

    def test_cuda_failure_has_cpu_fallback(self):
        code = open(portable.__file__, "r", encoding="utf-8").read()
        self.assertIn("自动回退CPU", code)
        self.assertIn("cpu_load()", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
