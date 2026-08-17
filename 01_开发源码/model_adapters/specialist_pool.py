from pathlib import Path
import json


REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "ai_models"
    / "specialist_asset_registry.json"
)


class SpecialistAsset:

    def __init__(self, item):
        self.data=item
        self.model_id=item["model_id"]
        self.task=item["task"]
        self.root=Path(item["root"])
        self.status=item["status"]
        self.model=None

    def validate_assets(self):
        return (
            self.root.exists()
            and self.data["weight_files"] > 0
        )

    def unload(self):
        self.model=None


items=json.loads(
    REGISTRY.read_text(encoding="utf-8")
)

SPECIALIST_POOL={
    x["model_id"]:SpecialistAsset(x)
    for x in items
}
