from ai_models.component_weight_loader import (
    ComponentWeightLoader,
)


class VLMComponentAdapter:

    def __init__(self, component_id):
        self.component_id=component_id
        self.loader=ComponentWeightLoader()
        self.status="REGISTERED_UNTESTED"

    def describe(self):
        return self.loader.describe(
            self.component_id
        )

    def unload(self):
        pass


COMPONENT_IDS = [
    "MedGemma-27B::vision_encoder",
    "MedGemma-27B::projector",

    "M3D-LaMed-Phi-3-4B::vision_encoder",
    "M3D-LaMed-Phi-3-4B::projector",
    "M3D-LaMed-Phi-3-4B::segmentation_head",

    "14_MAIRA_2_ModelScope::vision_encoder",
    "14_MAIRA_2_ModelScope::projector",

    "Fleming-VL-8B::vision_encoder",

    "Lingshu-I-8B::vision_encoder",
    "Lingshu-I-8B::projector",

    "LLaVA-Med-7B::vision_encoder",
    "LLaVA-Med-7B::projector",

    "Hulu-Med-4B::vision_encoder",
    "Hulu-Med-4B::projector",

    "MedGemma-1.5-4B::vision_encoder",
    "MedGemma-1.5-4B::projector",

    "MedGemma-4B-old::vision_encoder",
    "MedGemma-4B-old::projector",
]


VLM_COMPONENT_POOL = {}

for cid in COMPONENT_IDS:
    try:
        VLM_COMPONENT_POOL[cid] = (
            VLMComponentAdapter(cid)
        )
    except Exception:
        pass
