from .case_router import select_models
from .result_fusion import fuse_results
from .report_generator import generate_report

from output.lesion_overlay import build_overlays


class PhoenixPipeline:

    def __init__(self, model_hub):
        self.model_hub = model_hub

    def analyze(self, case):
        selected = select_models(case)

        self.model_hub.load_selected(
            selected
        )

        raw = self.model_hub.predict_selected(
            case,
            selected,
        )

        incomplete = []

        for name in selected:
            if self.model_hub.status.get(name) != "loaded":
                incomplete.append(name)
                continue

            data = raw.get(name)

            if isinstance(data, dict) and "error" in data:
                incomplete.append(name)

        result = fuse_results(raw)

        result = generate_report(
            result,
            incomplete,
        )

        return {
            "case_id": case.case_id,
            "selected_models": selected,
            "incomplete_models": incomplete,
            "analysis": result,
            "overlays": build_overlays(
                result.lesions
            ),
        }
