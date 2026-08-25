from __future__ import annotations

"""Production SSD storage routing with explicit-path isolation.

The portable first-level SSD folder is used only for the canonical Phoenix
project runtime. Explicit WorkbenchPaths supplied by tests, migrations or
sandbox tools retain their own evidence/runtime roots and cannot leak files into
another task or test process.
"""

from pathlib import Path

_INSTALLED = False


def _same_root(left, right) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from .config import get_paths
    from .office_translation import OfficeDocumentTranslator
    from .pdf_assets import PDFAssetStore
    from .translation_learning_collector import TranslationLearningCollector
    from .translation_learning_pool import TranslationLearningPool
    from .translator import PDFTranslator
    from . import translation_ssd_storage as storage_mod

    canonical = get_paths()
    root = storage_mod.translation_storage_root(canonical)
    storage_mod._ensure_layout(root)
    stats = storage_mod._migrate_legacy_translation_data(canonical, root, force=False)
    storage_mod._print_migration(root, stats)

    original_pdf_init = PDFTranslator.__init__
    original_office_init = OfficeDocumentTranslator.__init__
    original_learning_init = TranslationLearningPool.__init__
    original_collector_init = TranslationLearningCollector.__init__

    def _is_primary(paths) -> bool:
        return _same_root(getattr(paths, "project_root", ""), canonical.project_root)

    def pdf_init(self, paths, llm):
        original_pdf_init(self, paths, llm)
        if not _is_primary(paths):
            return
        portable = storage_mod.translation_storage_root(paths)
        storage_mod._migrate_legacy_translation_data(paths, portable)
        self.output_root = portable / "PDF整本翻译"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.assets = PDFAssetStore(portable / "_运行缓存")

    def office_init(self, paths, engine):
        original_office_init(self, paths, engine)
        if not _is_primary(paths):
            return
        portable = storage_mod.translation_storage_root(paths)
        storage_mod._migrate_legacy_translation_data(paths, portable)
        self.output_root = portable / "同格式医学翻译"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def learning_init(self, root=None):
        if root is not None:
            original_learning_init(self, root)
            return
        original_learning_init(
            self,
            storage_mod.translation_storage_root(canonical) / "人工修订与学习",
        )

    def collector_init(self, root=None):
        if root is not None:
            original_collector_init(self, root)
            return
        original_collector_init(
            self,
            storage_mod.translation_storage_root(canonical) / "人工修订与学习",
        )

    PDFTranslator.__init__ = pdf_init
    OfficeDocumentTranslator.__init__ = office_init
    TranslationLearningPool.__init__ = learning_init
    TranslationLearningCollector.__init__ = collector_init

    _INSTALLED = True
