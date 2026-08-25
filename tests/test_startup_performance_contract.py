from __future__ import annotations

import types
import unittest


class _Label:
    def __init__(self):
        self.text = ""
        self.tooltip = ""

    def setText(self, value):
        self.text = str(value)

    def setToolTip(self, value):
        self.tooltip = str(value)


class _StatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, value, *_args):
        self.message = str(value)


class StartupPerformanceContractTests(unittest.TestCase):
    def test_window_init_does_not_run_heavy_status_probes(self):
        from phoenix_knowledge import startup_performance_gui as perf

        class Window:
            def __init__(self):
                self.heavy_models = 0
                self.heavy_status = 0
                self.heavy_compute = 0
                self.translation_models_label = _Label()
                self.compute_status_label = _Label()
                self._bar = _StatusBar()
                self.refresh_translation_models()
                self._status_text()
                self._update_compute_label()

            def refresh_translation_models(self):
                self.heavy_models += 1

            def _status_text(self):
                self.heavy_status += 1
                return "heavy"

            def _update_compute_label(self):
                self.heavy_compute += 1

            def statusBar(self):
                return self._bar

        gui_module = types.SimpleNamespace(WorkbenchWindow=Window)
        perf._INSTALLED = False
        perf.install(gui_module)

        window = gui_module.WorkbenchWindow()
        self.assertEqual(window.heavy_models, 0)
        self.assertEqual(window.heavy_status, 0)
        self.assertEqual(window.heavy_compute, 0)
        self.assertIn("按需检测", window.translation_models_label.text)
        self.assertIn("按需检测", window.compute_status_label.text)

        window.refresh_translation_models()
        self.assertEqual(window.heavy_models, 1)

    def test_remote_status_does_not_require_cuda_probe(self):
        from phoenix_knowledge.startup_performance_gui import _remote_status_without_cuda

        class Gateway:
            _warning = ""

            def remote_url(self):
                return "https://api.deepseek.com"

            def remote_allowed(self):
                return True

            def remote_is_public(self):
                return True

            def remote_api_key(self):
                return "test-key"

            def remote_model(self, _profile=None):
                return "deepseek-v4-pro"

            def provider_label(self):
                return "DeepSeek"

        status = _remote_status_without_cuda(Gateway())
        self.assertEqual(status.effective_mode, "remote")
        self.assertFalse(status.cuda_available)
        self.assertEqual(status.gpu_count, 0)
        self.assertEqual(status.warning, "")


if __name__ == "__main__":
    unittest.main()
