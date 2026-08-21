from __future__ import annotations

import os
import subprocess
import sys
import unittest


class FinalGUIRuntimeContractTests(unittest.TestCase):
    def test_fresh_process_installs_final_busy_guards(self):
        script = r'''
from phoenix_knowledge import gui as gui_module
from phoenix_knowledge.gui_bootstrap import GUI_INSTALL_ORDER, install_gui_stack

applied = install_gui_stack(gui_module, strict=True)
assert tuple(applied) == GUI_INSTALL_ORDER, applied
cls = gui_module.WorkbenchWindow
assert getattr(cls, "__phoenix_gui_contract__", 0) >= 3
assert getattr(cls, "__phoenix_release_gui_hardening__", 0) >= 2
for name in (
    "add_pdfs",
    "add_documents",
    "ask_question",
    "build_embeddings",
    "start_organize",
    "resume_organize",
    "start_translation",
    "start_notes_organize",
):
    method = getattr(cls, name, None)
    assert callable(method), name
    assert getattr(method, "__phoenix_busy_guard__", False), name
print("FINAL_GUI_RUNTIME_CONTRACT=PASS")
'''
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            timeout=90,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(completed.stdout + "\n" + completed.stderr),
        )
        self.assertIn("FINAL_GUI_RUNTIME_CONTRACT=PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
