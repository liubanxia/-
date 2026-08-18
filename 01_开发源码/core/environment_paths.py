from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PhoenixEnvironmentPaths:
    project_root: Path
    image_root: Optional[Path]
    environment_name: str


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def _normalize(path: str | os.PathLike | None) -> Optional[Path]:
    if not path:
        return None
    try:
        return Path(path).expanduser().resolve()
    except Exception:
        return Path(path).expanduser()


def resolve_project_root(explicit: str | os.PathLike | None = None) -> Path:
    env_value = os.environ.get("PHOENIX_PROJECT_ROOT")
    candidates = []

    for value in (explicit, env_value):
        p = _normalize(value)
        if p is not None:
            candidates.append(p)

    candidates.extend(
        [
            Path(r"G:\project_phoenix"),  # 医院 SSD
            Path(r"D:\project_phoenix"),  # 网吧 SSD
        ]
    )

    existing = _existing(candidates)
    if existing:
        return existing[0]

    # 代码自身位于 <root>/01_开发源码/core/environment_paths.py
    return Path(__file__).resolve().parents[2]


def resolve_image_root(explicit: str | os.PathLike | None = None) -> Optional[Path]:
    env_value = os.environ.get("PHOENIX_IMAGE_ROOT")

    for value in (explicit, env_value):
        p = _normalize(value)
        if p is not None and p.exists():
            return p

    hospital_candidates = [
        Path(r"D:\YUNPACS\放射诊断\ImageDir_r"),
        Path(r"D:\YUNPACS"),
    ]

    existing = _existing(hospital_candidates)
    return existing[0] if existing else None


def detect_environment(project_root: Path) -> str:
    drive = project_root.drive.upper()
    if drive == "G:":
        return "hospital"
    if drive == "D:":
        return "internet_cafe"
    return "portable"


def get_environment_paths(
    project_root: str | os.PathLike | None = None,
    image_root: str | os.PathLike | None = None,
) -> PhoenixEnvironmentPaths:
    root = resolve_project_root(project_root)
    images = resolve_image_root(image_root)
    return PhoenixEnvironmentPaths(
        project_root=root,
        image_root=images,
        environment_name=detect_environment(root),
    )
