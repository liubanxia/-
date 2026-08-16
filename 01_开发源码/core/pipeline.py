from .result_fusion import fuse_results
from .report_generator import generate_report
from output.lesion_overlay import build_overlays


class PhoenixPipeline:

    def __init__(self, model_hub):
        self.model_hub = model_hub

    def analyze(self, case):
        raw = self.model_hub.predict_all(case)

        result = fuse_results(raw)
        result = generate_report(result)

        overlays = build_overlays(
            result.lesions
        )

        return {
            "case_id": case.case_id,
            "analysis": result,
            "overlays": overlays,
        }
