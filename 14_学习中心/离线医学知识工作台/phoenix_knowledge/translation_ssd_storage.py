from __future__ import annotations

import os
import shutil
from pathlib import Path


_FOLDER = "知识平台翻译"
_ENV = "PHOENIX_TRANSLATION_STORAGE_ROOT"
_INSTALLED = False
_MIGRATED: set[str] = set()


def translation_storage_root(paths=None) -> Path:
    """Return the portable first-level translation folder on the Phoenix SSD."""

    override = os.environ.get(_ENV, "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        if paths is not None:
            project_root = Path(paths.project_root).resolve()
        else:
            from .config import discover_project_root

            project_root = discover_project_root().resolve()

        # On Windows the project may remount as D:, G:, etc. Always derive the
        # current drive from project_root so the SSD stays portable.
        if os.name == "nt" and project_root.drive:
            root = Path(project_root.drive + "\\") / _FOLDER
        else:
            # Test/non-Windows fallback: keep the folder beside the project.
            root = project_root.parent / _FOLDER

    root.mkdir(parents=True, exist_ok=True)
    return root


def _merge_move(source: Path, destination: Path) -> None:
    """Move legacy Phoenix-owned data without overwriting newer SSD data."""

    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        target = destination / source.name
        if not target.exists():
            shutil.move(str(source), str(target))
        return

    for item in tuple(source.iterdir()):
        target = destination / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                _merge_move(item, target)
            continue
        try:
            shutil.move(str(item), str(target))
        except OSError:
            # A locked preview/cache file must never prevent Phoenix startup.
            continue
    try:
        source.rmdir()
    except OSError:
        pass


def _migrate_legacy_translation_data(paths, root: Path) -> None:
    key = str(Path(paths.project_root).resolve()).casefold()
    if key in _MIGRATED:
        return
    _MIGRATED.add(key)

    evidence = Path(paths.evidence_root)
    runtime = Path(paths.runtime_root)
    project = Path(paths.project_root)

    _merge_move(evidence / "PDF整本翻译", root / "PDF整本翻译")
    _merge_move(evidence / "同格式医学翻译", root / "同格式医学翻译")

    # PDF extraction assets are translation runtime data too. Move them out of
    # the project tree so every newly written translation artifact lives under
    # the SSD first-level translation folder.
    _merge_move(runtime / "pdf_assets", root / "_运行缓存" / "pdf_assets")

    # Adopt old/default learning pools when present. Reviewed corrections are
    # preserved; this migration does not train or alter model weights.
    learning = root / "人工修订与学习"
    for old in (
        project / "translation_learning_pool",
        runtime / "translation_learning_pool",
        evidence / "translation_learning_pool",
    ):
        _merge_move(old, learning)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .office_translation import OfficeDocumentTranslator
    from .pdf_assets import PDFAssetStore
    from .translation_learning_pool import TranslationLearningPool
    from .translator import PDFTranslator

    original_pdf_init = PDFTranslator.__init__
    original_office_init = OfficeDocumentTranslator.__init__
    original_learning_init = TranslationLearningPool.__init__

    def pdf_init(self, paths, llm):
        original_pdf_init(self, paths, llm)
        root = translation_storage_root(paths)
        _migrate_legacy_translation_data(paths, root)
        self.output_root = root / "PDF整本翻译"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.assets = PDFAssetStore(root / "_运行缓存")

    def office_init(self, paths, engine):
        original_office_init(self, paths, engine)
        root = translation_storage_root(paths)
        _migrate_legacy_translation_data(paths, root)
        self.output_root = root / "同格式医学翻译"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def learning_init(self, root=None):
        if root is None:
            root = translation_storage_root() / "人工修订与学习"
        original_learning_init(self, root)

    PDFTranslator.__init__ = pdf_init
    OfficeDocumentTranslator.__init__ = office_init
    TranslationLearningPool.__init__ = learning_init

    # Create the complete first-level layout immediately. This also makes the
    # destination visible in Explorer before the first translation starts.
    root = translation_storage_root()
    for child in (
        "PDF整本翻译",
        "同格式医学翻译",
        "人工修订与学习",
        "_运行缓存",
    ):
        (root / child).mkdir(parents=True, exist_ok=True)

    print(
        f"[Phoenix][翻译存储] SSD一级目录已启用：{root}",
        flush=True,
    )
