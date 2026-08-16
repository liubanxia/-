from .case_router import select_models
from .result_fusion import fuse_results
from .report_generator import generate_report

from output.lesion_overlay import build_overlays


class PhoenixPipeline:

    def __init__(self, model_hub):
        self.model_hub = model_hub

    def analyze(self, case):
        selected = select_models(case)

        self.model_hub.load_selected(selected)

        raw = self.model_hub.predict_selected(
            case,
            selected,
        )

        result = fuse_results(raw)
        result = generate_report(result)

        return {
            "case_id": case.case_id,
            "selected_models": selected,
            "analysis": result,
            "overlays": build_overlays(result.lesions),
        }
