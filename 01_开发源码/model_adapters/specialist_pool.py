from __future__ import annotations

import json
from pathlib import Path

from core.environment_paths import resolve_project_root


REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "ai_models"
    / "specialist_asset_registry.json"
)


def _resolve_asset_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_project_root() / path


class SpecialistAsset:

    def __init__(self, item):
        self.data = dict(item)
        self.model_id = item["model_id"]
        self.task = item["task"]
        self.root = _resolve_asset_path(item["root"])
        self.status = item.get("status", "REGISTERED_UNTESTED")
        self.model = None

    @property
    def weight_paths(self):
        return [
            _resolve_asset_path(value)
            for value in self.data.get("weights", [])
        ]

    def validate_assets(self):
        if not self.root.exists():
            return False

        weights = self.weight_paths
        if not weights:
            return False

        return all(path.exists() for path in weights)

    def unload(self):
        self.model = None


items = json.loads(
    REGISTRY.read_text(encoding="utf-8")
)

SPECIALIST_POOL = {
    x["model_id"]: SpecialistAsset(x)
    for x in items
}
