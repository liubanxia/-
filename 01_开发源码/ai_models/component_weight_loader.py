from pathlib import Path
import json


class ComponentWeightLoader:

    def __init__(self, specs_path=None):

        if specs_path is None:
            specs_path=(
                Path(__file__).resolve().parent
                / "phoenix_component_specs.json"
            )

        self.specs=json.loads(
            Path(specs_path).read_text(
                encoding="utf-8"
            )
        )

    def get_spec(self, component_id):
        for x in self.specs:
            if x["component_id"]==component_id:
                return x
        raise KeyError(component_id)

    def select_keys(self, component_id):

        spec=self.get_spec(component_id)
        root=Path(spec["source_path"])
        terms=[
            x.lower()
            for x in spec["match_terms"]
        ]

        keys=[]

        for idx in root.glob(
            "*safetensors.index.json"
        ):
            data=json.loads(
                idx.read_text(encoding="utf-8")
            )

            for k in data.get(
                "weight_map",{}
            ):
                low=k.lower()

                if any(t in low for t in terms):
                    keys.append(k)

        if not keys:
            try:
                from safetensors import safe_open

                for f in root.glob(
                    "*.safetensors"
                ):
                    with safe_open(
                        str(f),
                        framework="pt",
                        device="cpu",
                    ) as sf:

                        for k in sf.keys():
                            low=k.lower()

                            if any(
                                t in low
                                for t in terms
                            ):
                                keys.append(k)
            except Exception:
                pass

        return sorted(set(keys))

    def describe(self, component_id):

        spec=self.get_spec(component_id)

        return {
            **spec,
            "matched_tensor_keys":
                len(
                    self.select_keys(
                        component_id
                    )
                ),
        }
