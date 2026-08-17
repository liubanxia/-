from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

SPECIALIST_REGISTRY = (
    Path(__file__).resolve().parent
    / "specialist_asset_registry.json"
)

COMPONENT_REGISTRY = (
    Path(__file__).resolve().parent
    / "phoenix_component_registry.json"
)


class PhoenixExpertCatalog:
    def __init__(self):
        self.experts = {}
        self._load_components()
        self._load_specialists()

    def _load_components(self):
        if not COMPONENT_REGISTRY.exists():
            return

        data = json.loads(
            COMPONENT_REGISTRY.read_text(encoding="utf-8")
        )

        for item in data:
            name = item.get("component_id")
            if not name:
                continue

            self.experts[name] = {
                **item,
                "integration_status": "INTEGRATED_UNTESTED",
            }

    def _load_specialists(self):
        if not SPECIALIST_REGISTRY.exists():
            return

        data = json.loads(
            SPECIALIST_REGISTRY.read_text(encoding="utf-8")
        )

        for item in data:
            name = item.get("model_id")
            if not name:
                continue

            self.experts[name] = {
                **item,
                "integration_status": "INTEGRATED_UNTESTED",
            }

    def get(self, name):
        return self.experts.get(name)

    def all(self):
        return dict(self.experts)

    def by_task(self, task):
        return {
            k: v
            for k, v in self.experts.items()
            if v.get("task") == task
            or v.get("component_type") == task
        }


EXPERT_CATALOG = PhoenixExpertCatalog()
