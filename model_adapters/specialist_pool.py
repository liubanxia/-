from __future__ import annotations

import json
from pathlib import Path


class SpecialistAsset:
    def __init__(self, item, project_root=None):
        self.data = dict(item)
        self.model_id = item["model_id"]
        self.task = item["task"]
        self.project_root = Path(project_root) if project_root else Path.cwd()
        root = Path(item["root"])
        self.root = root if root.is_absolute() else self.project_root / root
        self.status = item.get("status", "REGISTERED_UNTESTED")
        self.model = None

    @property
    def weight_paths(self):
        out = []
        for value in self.data.get("weights", []):
            path = Path(value)
            out.append(path if path.is_absolute() else self.project_root / path)
        return out

    def validate_assets(self):
        weights = self.weight_paths
        return self.root.exists() and bool(weights) and all(path.exists() for path in weights)

    def unload(self):
        self.model = None


def load_specialist_pool(registry_path=None, project_root=None):
    registry = Path(registry_path) if registry_path else Path(__file__).resolve().parents[1] / "ai_models" / "specialist_asset_registry.json"
    if not registry.exists():
        return {}
    items = json.loads(registry.read_text(encoding="utf-8"))
    return {item["model_id"]: SpecialistAsset(item, project_root=project_root) for item in items}


SPECIALIST_POOL = load_specialist_pool()
