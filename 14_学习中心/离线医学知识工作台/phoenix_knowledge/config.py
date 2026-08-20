from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def discover_project_root(start: Path | None = None) -> Path:
    start = Path(start or __file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "01_开发源码").exists() or (candidate / ".git").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def model_dir_ready(path: Path) -> bool:
    """A model directory is READY only when config and real weights exist."""

    path = Path(path)
    try:
        if not path.is_dir() or not (path / "config.json").is_file():
            return False
        direct_weights = (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if any(
            (path / name).is_file()
            and (path / name).stat().st_size > 0
            for name in direct_weights
        ):
            return True
        # Sharded downloads can use nonstandard shard filenames while the
        # index/config remain conventional.
        return any(
            item.is_file()
            and item.stat().st_size > 0
            and item.suffix.lower() in {".safetensors", ".bin"}
            for item in path.iterdir()
        )
    except OSError:
        return False


def _write_pointer(pointer: Path, resolved: Path) -> None:
    try:
        temp = pointer.with_suffix(pointer.suffix + ".tmp")
        temp.write_text(str(resolved), encoding="utf-8")
        os.replace(temp, pointer)
    except OSError:
        pass


def _rebase_modelscope_pointer(model_root: Path, stale: Path) -> Path | None:
    """Recover a stale absolute ModelScope pointer after D:/G: remounting."""

    model_root = Path(model_root).resolve()
    cache_root = model_root / "_modelscope_cache"
    candidates: list[Path] = []

    stale_parts = list(stale.parts)
    try:
        marker_index = next(
            index
            for index, part in enumerate(stale_parts)
            if part == "_modelscope_cache"
        )
        relative = Path(*stale_parts[marker_index + 1 :])
        if relative.parts:
            candidates.append(cache_root / relative)
    except StopIteration:
        pass

    # Pointers from older Phoenix builds can contain the full project path.
    # Reconstruct the stable suffix below 04_AI模型 on the current SSD.
    try:
        marker_index = next(
            index
            for index, part in enumerate(stale_parts)
            if part == "04_AI模型"
        )
        project_root = model_root.parent.parent
        relative = Path(*stale_parts[marker_index:])
        candidates.append(project_root / relative)
    except StopIteration:
        pass

    for candidate in candidates:
        try:
            if candidate.is_dir() and model_dir_ready(candidate):
                return candidate.resolve()
        except OSError:
            continue

    # Only search the Phoenix-owned cache, and only when the old absolute
    # pointer is already broken. This keeps normal startup cheap.
    if cache_root.is_dir() and stale.name:
        try:
            for candidate in cache_root.rglob(stale.name):
                if candidate.is_dir() and model_dir_ready(candidate):
                    return candidate.resolve()
        except OSError:
            pass
    return None


def resolve_model_dir(model_root: Path, folder: str) -> Path:
    """Resolve a direct model folder or portable ModelScope cache pointer.

    Older Phoenix/ModelScope combinations stored an absolute snapshot path in
    ``MODELSCOPE_CACHE_PATH.txt``. The same physical SSD can be D: on a
    development PC and G: at the hospital, so a stale drive letter is rebased
    onto the current Phoenix-owned cache before the model is declared missing.
    """

    model_root = Path(model_root)
    target = model_root / folder
    pointer = target / "MODELSCOPE_CACHE_PATH.txt"

    if pointer.is_file():
        try:
            raw = pointer.read_text(encoding="utf-8").strip()
            if raw:
                stale = Path(raw).expanduser()
                if stale.is_dir() and model_dir_ready(stale):
                    return stale.resolve()
                recovered = _rebase_modelscope_pointer(model_root, stale)
                if recovered is not None:
                    _write_pointer(pointer, recovered)
                    return recovered
        except OSError:
            pass

    return target


@dataclass(frozen=True)
class WorkbenchPaths:
    project_root: Path
    source_root: Path
    runtime_root: Path
    evidence_root: Path
    model_root: Path
    database: Path
    structure_root: Path

    def ensure(self) -> "WorkbenchPaths":
        for path in (
            self.source_root,
            self.runtime_root,
            self.evidence_root,
            self.model_root,
            self.structure_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(project_root: Path | None = None) -> WorkbenchPaths:
    root = Path(
        os.environ.get("PHOENIX_PROJECT_ROOT")
        or project_root
        or discover_project_root()
    ).resolve()

    source_root = Path(
        os.environ.get("PHOENIX_KNOWLEDGE_SOURCE_ROOT")
        or root / "14_学习中心" / "PDF资料"
    )
    runtime_root = Path(
        os.environ.get("PHOENIX_KNOWLEDGE_RUNTIME_ROOT")
        or root / "14_学习中心" / "离线医学知识工作台_data"
    )
    evidence_root = Path(
        os.environ.get("PHOENIX_KNOWLEDGE_EVIDENCE_ROOT")
        or root / "15_证据中心" / "PDF知识整理"
    )
    model_root = Path(
        os.environ.get("PHOENIX_KNOWLEDGE_MODEL_ROOT")
        or root / "04_AI模型" / "知识工作台"
    )

    return WorkbenchPaths(
        project_root=root,
        source_root=source_root,
        runtime_root=runtime_root,
        evidence_root=evidence_root,
        model_root=model_root,
        database=runtime_root / "knowledge.sqlite3",
        structure_root=runtime_root / "docling_structure",
    ).ensure()
