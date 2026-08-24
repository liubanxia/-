from __future__ import annotations

import unittest

from phoenix_knowledge.translation_portable_model3_runtime import choose_device


class PortableModel3RuntimeTest(unittest.TestCase):
    def test_cpu_is_used_when_cuda_is_absent(self):
        self.assertEqual(
            choose_device(
                requested="auto",
                cuda_available=False,
                capability_major=None,
                probe_ok=False,
            ),
            "cpu",
        )

    def test_old_cuda_device_falls_back_to_cpu(self):
        self.assertEqual(
            choose_device(
                requested="auto",
                cuda_available=True,
                capability_major=3,
                probe_ok=True,
            ),
            "cpu",
        )

    def test_compatible_cuda_is_used_without_gpu_model_whitelist(self):
        self.assertEqual(
            choose_device(
                requested="auto",
                cuda_available=True,
                capability_major=8,
                probe_ok=True,
            ),
            "cuda:0",
        )

    def test_failed_cuda_probe_falls_back_to_cpu(self):
        self.assertEqual(
            choose_device(
                requested="auto",
                cuda_available=True,
                capability_major=8,
                probe_ok=False,
            ),
            "cpu",
        )

    def test_explicit_cpu_overrides_cuda(self):
        self.assertEqual(
            choose_device(
                requested="cpu",
                cuda_available=True,
                capability_major=9,
                probe_ok=True,
            ),
            "cpu",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
