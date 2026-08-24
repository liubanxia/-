from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


_FOLDER = "知识平台翻译"
_ENV = "PHOENIX_TRANSLATION_STORAGE_ROOT"
_INSTALLED = False
_MIGRATED: set[str] = set()


@dataclass
class MigrationStats:
    moved_files: int = 0
    moved_dirs: int = 0
    duplicate_files_removed: int = 0
    conflict_files_preserved: int = 0
    errors: int = 0

    def add(self, other: "MigrationStats") -> "MigrationStats":
        self.moved_files += int(other.moved_files)
        self.moved_dirs += int(other.moved_dirs)
        self.duplicate_files_removed += int(other.duplicate_files_removed)
        self.conflict_files_preserved += int(other.conflict_files_preserved)
        self.errors += int(other.errors)
        return self


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

        # The same SSD is D: on the development computer and G: at hospital.
        # Derive the root from the current project drive instead of hard-coding.
        if os.name == "nt" and project_root.drive:
            root = Path(project_root.drive + "\\") / _FOLDER
        else:
            root = project_root.parent / _FOLDER

    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _same_file(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        return _sha256(left) == _sha256(right)
    except OSError:
        return False


def _conflict_target(path: Path) -> Path:
    """Preserve both files when legacy and new locations contain different data."""

    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}__旧位置冲突{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法生成冲突保留文件名：{path}")


def _move_file(source: Path, target: Path) -> MigrationStats:
    stats = MigrationStats()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.move(str(source), str(target))
            stats.moved_files += 1
            return stats

        if target.is_file() and _same_file(source, target):
            source.unlink(missing_ok=True)
            stats.duplicate_files_removed += 1
            return stats

        conflict = _conflict_target(target)
        shutil.move(str(source), str(conflict))
        stats.moved_files += 1
        stats.conflict_files_preserved += 1
        return stats
    except OSError:
        stats.errors += 1
        return stats


def _merge_move(source: Path, destination: Path) -> MigrationStats:
    """Move legacy Phoenix translation data without losing conflicting files."""

    stats = MigrationStats()
    source = Path(source)
    destination = Path(destination)

    try:
        if not source.exists():
            return stats
        if source.resolve() == destination.resolve():
            return stats
    except OSError:
        pass

    if source.is_file():
        return _move_file(source, destination / source.name)

    destination.mkdir(parents=True, exist_ok=True)
    try:
        items = tuple(source.iterdir())
    except OSError:
        stats.errors += 1
        return stats

    for item in items:
        target = destination / item.name
        if item.is_dir():
            if target.exists() and target.is_file():
                try:
                    conflict = _conflict_target(target)
                    shutil.move(str(item), str(conflict))
                    stats.moved_dirs += 1
                    stats.conflict_files_preserved += 1
                except OSError:
                    stats.errors += 1
                continue

            before_exists = target.exists()
            child = _merge_move(item, target)
            stats.add(child)
            if not before_exists and target.exists():
                stats.moved_dirs += 1
            continue

        stats.add(_move_file(item, target))

    try:
        source.rmdir()
    except OSError:
        pass
    return stats


def _legacy_translation_sources(paths, root: Path) -> tuple[tuple[Path, Path], ...]:
    evidence = Path(paths.evidence_root)
    runtime = Path(paths.runtime_root)
    project = Path(paths.project_root)

    pairs: list[tuple[Path, Path]] = [
        (evidence / "PDF整本翻译", root / "PDF整本翻译"),
        (evidence / "同格式医学翻译", root / "同格式医学翻译"),
        (runtime / "pdf_assets", root / "_运行缓存" / "pdf_assets"),
        (project / "translation_learning_pool", root / "人工修订与学习"),
        (runtime / "translation_learning_pool", root / "人工修订与学习"),
        (evidence / "translation_learning_pool", root / "人工修订与学习"),
    ]

    # Older Phoenix versions used several translation/cache folder names. Move
    # translation-specific entries discovered directly under evidence/runtime,
    # while deliberately leaving the knowledge database and unrelated learning
    # data in place.
    for parent, destination in (
        (evidence, root / "历史翻译数据"),
        (runtime, root / "_运行缓存"),
    ):
        if not parent.is_dir():
            continue
        try:
            children = tuple(parent.iterdir())
        except OSError:
            continue
        for child in children:
            name = child.name.casefold()
            if child in {item[0] for item in pairs}:
                continue
            if "翻译" in child.name or name.startswith("translation_"):
                pairs.append((child, destination / child.name))

    # Deduplicate source paths while retaining deterministic order.
    result: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for source, destination in pairs:
        key = str(source).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append((source, destination))
    return tuple(result)


def _migrate_legacy_translation_data(
    paths,
    root: Path,
    *,
    force: bool = False,
) -> MigrationStats:
    key = str(Path(paths.project_root).resolve()).casefold()
    if key in _MIGRATED and not force:
        return MigrationStats()

    stats = MigrationStats()
    for source, destination in _legacy_translation_sources(paths, root):
        stats.add(_merge_move(source, destination))

    _MIGRATED.add(key)
    return stats


def _ensure_layout(root: Path) -> None:
    for child in (
        "PDF整本翻译",
        "同格式医学翻译",
        "人工修订与学习",
        "历史翻译数据",
        "_运行缓存",
    ):
        (root / child).mkdir(parents=True, exist_ok=True)


def _print_migration(root: Path, stats: MigrationStats) -> None:
    print(f"[Phoenix][翻译存储] SSD一级目录：{root}", flush=True)
    print(
        "[Phoenix][翻译存储] 旧数据迁移完成 | "
        f"文件={stats.moved_files} | "
        f"目录={stats.moved_dirs} | "
        f"重复清理={stats.duplicate_files_removed} | "
        f"冲突保留={stats.conflict_files_preserved} | "
        f"错误={stats.errors}",
        flush=True,
    )


def migrate_now(*, force: bool = True) -> tuple[Path, MigrationStats]:
    """Physically migrate existing translation data immediately."""

    from .config import get_paths

    paths = get_paths()
    root = translation_storage_root(paths)
    _ensure_layout(root)
    stats = _migrate_legacy_translation_data(paths, root, force=force)
    _print_migration(root, stats)
    return root, stats


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .config import get_paths
    from .office_translation import OfficeDocumentTranslator
    from .pdf_assets import PDFAssetStore
    from .translation_learning_collector import TranslationLearningCollector
    from .translation_learning_pool import TranslationLearningPool
    from .translator import PDFTranslator

    original_pdf_init = PDFTranslator.__init__
    original_office_init = OfficeDocumentTranslator.__init__
    original_learning_init = TranslationLearningPool.__init__
    original_collector_init = TranslationLearningCollector.__init__

    # IMPORTANT: migrate during package startup, not only after the user starts
    # a PDF/Office translation. This was the missing trigger in the first SSD
    # storage implementation.
    paths = get_paths()
    root = translation_storage_root(paths)
    _ensure_layout(root)
    stats = _migrate_legacy_translation_data(paths, root, force=False)
    _print_migration(root, stats)

    def pdf_init(self, paths, llm):
        original_pdf_init(self, paths, llm)
        storage = translation_storage_root(paths)
        _migrate_legacy_translation_data(paths, storage)
        self.output_root = storage / "PDF整本翻译"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.assets = PDFAssetStore(storage / "_运行缓存")

    def office_init(self, paths, engine):
        original_office_init(self, paths, engine)
        storage = translation_storage_root(paths)
        _migrate_legacy_translation_data(paths, storage)
        self.output_root = storage / "同格式医学翻译"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def learning_init(self, root=None):
        storage = translation_storage_root() / "人工修订与学习"
        original_learning_init(self, storage)

    def collector_init(self, root=None):
        storage = translation_storage_root() / "人工修订与学习"
        original_collector_init(self, storage)

    PDFTranslator.__init__ = pdf_init
    OfficeDocumentTranslator.__init__ = office_init
    TranslationLearningPool.__init__ = learning_init
    TranslationLearningCollector.__init__ = collector_init


if __name__ == "__main__":
    migrate_now(force=True)
