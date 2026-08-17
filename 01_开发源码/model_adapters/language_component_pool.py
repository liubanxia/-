from ai_models.component_weight_loader import ComponentWeightLoader


class LanguageComponent:
    def __init__(self, component_id):
        self.component_id = component_id
        self.loader = ComponentWeightLoader()
        self.status = "REGISTERED_UNTESTED"

    def describe(self):
        return self.loader.describe(self.component_id)


LANGUAGE_IDS = [
    "Lingshu-32B::language_model",
    "MedGemma-27B::language_model",
    "M3D-LaMed-Phi-3-4B::language_model",
    "14_MAIRA_2_ModelScope::language_model",
    "HealthGPT-Pro-8B::language_model",
    "Lingshu-7B::language_model",
    "Fleming-VL-8B::language_model",
    "Lingshu-I-8B::language_model",
    "LLaVA-Med-7B::language_model",
    "Hulu-Med-4B::language_model",
    "HealthGPT-Pro-4B::language_model",
    "MedGemma-1.5-4B::language_model",
    "MedGemma-4B-old::language_model",
]

LANGUAGE_POOL = {}

for cid in LANGUAGE_IDS:
    try:
        LANGUAGE_POOL[cid] = LanguageComponent(cid)
    except Exception:
        pass
