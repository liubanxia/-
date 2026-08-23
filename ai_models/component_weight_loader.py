from pathlib import Path
import json


class ComponentWeightLoader:
    def __init__(self, specs_path=None):
        if specs_path is None:
            specs_path = Path(__file__).resolve().parent / "phoenix_component_specs.json"
        path = Path(specs_path)
        self.specs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def get_spec(self, component_id):
        for item in self.specs:
            if item.get("component_id") == component_id:
                return item
        raise KeyError(component_id)

    def select_keys(self, component_id):
        spec = self.get_spec(component_id)
        root = Path(spec["source_path"])
        terms = [x.lower() for x in spec.get("match_terms", [])]
        keys = []
        for idx in root.glob("*safetensors.index.json"):
            data = json.loads(idx.read_text(encoding="utf-8"))
            for key in data.get("weight_map", {}):
                if any(term in key.lower() for term in terms):
                    keys.append(key)
        if not keys:
            try:
                from safetensors import safe_open
                for file in root.glob("*.safetensors"):
                    with safe_open(str(file), framework="pt", device="cpu") as sf:
                        for key in sf.keys():
                            if any(term in key.lower() for term in terms):
                                keys.append(key)
            except Exception:
                pass
        return sorted(set(keys))

    def describe(self, component_id):
        spec = self.get_spec(component_id)
        return {**spec, "matched_tensor_keys": len(self.select_keys(component_id))}
