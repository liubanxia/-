from pathlib import Path
import json


class PhoenixComponentRegistry:

    def __init__(self, registry_path=None):

        if registry_path is None:
            registry_path = (
                Path(__file__).resolve().parent
                / "phoenix_component_registry.json"
            )

        self.path = Path(registry_path)

        self.components = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

    def all(self):
        return list(self.components)

    def by_type(self, component_type):
        return [
            x for x in self.components
            if x["component_type"]
            == component_type
        ]

    def by_model(self, model_name):
        return [
            x for x in self.components
            if x["source_model"]
            == model_name
        ]

    def get(self, component_id):
        for x in self.components:
            if x["component_id"] == component_id:
                return x
        return None
