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


def resolve_model_dir(model_root: Path, folder: str) -> Path:
    """Resolve either a direct model folder or a ModelScope cache pointer.

    Some ModelScope releases accept ``local_dir`` and place files exactly in
    Phoenix's model directory. Older releases only expose ``cache_dir`` and
    return their own snapshot path. ``model_download.py`` stores that returned
    path in ``MODELSCOPE_CACHE_PATH.txt`` so the offline runtime can still find
    the actual snapshot without moving multi-GB model files.
    """

    target = Path(model_root) / folder
    pointer = target / "MODELSCOPE_CACHE_PATH.txt"

    if pointer.is_file():
        try:
            raw = pointer.read_text(encoding="utf-8").strip()
            if raw:
                resolved = Path(raw).expanduser()
                if resolved.exists() and resolved.is_dir():
                    return resolved.resolve()
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
