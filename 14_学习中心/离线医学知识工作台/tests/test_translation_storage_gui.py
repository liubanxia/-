from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from phoenix_knowledge.config import WorkbenchPaths
from phoenix_knowledge.translation_layout_compact import LAYOUT_SOURCE_TRANSLATED
from phoenix_knowledge.translation_pdf import (
    LAYOUT_ORIGINAL_BILINGUAL,
    LAYOUT_TRANSLATED_ONLY,
)
from phoenix_knowledge.translation_storage_gui import _release_ratio_target
from phoenix_knowledge.workbench import MedicalKnowledgeWorkbench


_APP = None


def _app():
    global _APP
    if _APP is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _paths(root: Path) -> WorkbenchPaths:
    return WorkbenchPaths(
        project_root=root,
        source_root=root / "sources",
        runtime_root=root / "runtime",
        evidence_root=root / "evidence",
        model_root=root / "models",
        database=root / "runtime" / "knowledge.sqlite3",
        structure_root=root / "runtime" / "structure",
    ).ensure()


class TranslationStorageGuiTests(unittest.TestCase):
    def test_gui_defaults_to_compact_one_pdf_without_split_volumes(self):
        _app()
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"PHOENIX_KNOWLEDGE_ACCELERATOR": "cpu"},
            clear=False,
        ):
            from phoenix_knowledge import gui as gui_module
            from phoenix_knowledge.gui_enhancements import install as install_gui_enhancements
            from phoenix_knowledge.translation_storage_gui import install as install_storage_gui

            install_gui_enhancements(gui_module)
            install_storage_gui(gui_module)
            workbench = MedicalKnowledgeWorkbench(_paths(Path(temp)))
            with patch.object(
                gui_module,
                "MedicalKnowledgeWorkbench",
                return_value=workbench,
            ):
                window = gui_module.WorkbenchWindow()
            try:
                self.assertIs(window.workbench, workbench)
                self.assertTrue(hasattr(window, "translation_part_pages"))
                self.assertEqual(window.translation_part_pages.minimum(), 0)
                self.assertEqual(window.translation_part_pages.value(), 0)
                self.assertEqual(
                    window.translation_part_pages.specialValueText(),
                    "不生成分册",
                )
                self.assertEqual(
                    window.translation_layout_combo.currentData(),
                    LAYOUT_SOURCE_TRANSLATED,
                )
            finally:
                window.close()
                workbench.close()

    def test_release_ratio_targets_match_product_contract(self):
        self.assertEqual(_release_ratio_target(LAYOUT_SOURCE_TRANSLATED), 1.18)
        self.assertEqual(_release_ratio_target(LAYOUT_TRANSLATED_ONLY), 1.30)
        self.assertEqual(_release_ratio_target(LAYOUT_ORIGINAL_BILINGUAL), 1.50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
