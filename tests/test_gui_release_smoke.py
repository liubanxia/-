from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from phoenix_knowledge import gui as gui_module
from phoenix_knowledge.compute_gui import install as install_compute_gui
from phoenix_knowledge.document_gui import install as install_document_gui
from phoenix_knowledge.gui_enhancements import install as install_gui_enhancements


install_gui_enhancements(gui_module)
install_compute_gui(gui_module)
install_document_gui(gui_module)


class _BatchWorkbench:
    def ingest(self, path: Path, progress=None):
        if path.name.startswith("bad"):
            raise RuntimeError("simulated broken document")
        if progress:
            progress(1, 1, "done")
        return SimpleNamespace(
            pages_indexed=1,
            pages_total=1,
            warning="",
        )


class GUIReleaseSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_batch_ingest_continues_after_one_broken_file(self):
        worker = gui_module.IngestWorker(
            _BatchWorkbench(),
            ["bad.pdf", "good.pdf"],
        )
        completed: list[str] = []
        failed: list[str] = []
        worker.completed.connect(completed.append)
        worker.failed.connect(failed.append)
        worker.run()
        self.assertTrue(completed)
        self.assertFalse(failed)
        self.assertIn("成功 1，失败 1", completed[-1])
        self.assertIn("good.pdf", completed[-1])
        self.assertIn("bad.pdf", completed[-1])

    def test_full_gui_constructs_offscreen_with_latency_gate(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "PHOENIX_PROJECT_ROOT": temp,
                "PHOENIX_KNOWLEDGE_ACCELERATOR": "cpu",
                "PHOENIX_KNOWLEDGE_DEEP_QA": "0",
            },
            clear=False,
        ):
            started = time.perf_counter()
            window = gui_module.WorkbenchWindow()
            window.show()
            self.app.processEvents()
            elapsed = time.perf_counter() - started
            try:
                self.assertTrue(window.isVisible())
                self.assertLess(elapsed, 8.0)
            finally:
                window.close()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
