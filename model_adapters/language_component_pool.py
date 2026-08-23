from ai_models.component_weight_loader import ComponentWeightLoader


class LanguageComponent:
    def __init__(self, component_id, loader=None):
        self.component_id = component_id
        self.loader = loader or ComponentWeightLoader()
        self.status = "REGISTERED_UNTESTED"

    def describe(self):
        return self.loader.describe(self.component_id)


LANGUAGE_IDS = [
    "Lingshu-32B::language_model", "MedGemma-27B::language_model",
    "M3D-LaMed-Phi-3-4B::language_model", "14_MAIRA_2_ModelScope::language_model",
    "HealthGPT-Pro-8B::language_model", "Lingshu-7B::language_model",
    "Fleming-VL-8B::language_model", "Lingshu-I-8B::language_model",
    "LLaVA-Med-7B::language_model", "Hulu-Med-4B::language_model",
    "HealthGPT-Pro-4B::language_model", "MedGemma-1.5-4B::language_model",
]


def build_language_pool(loader=None):
    loader = loader or ComponentWeightLoader()
    available = {item.get("component_id") for item in loader.specs}
    return {cid: LanguageComponent(cid, loader=loader) for cid in LANGUAGE_IDS if cid in available}


LANGUAGE_POOL = build_language_pool()
