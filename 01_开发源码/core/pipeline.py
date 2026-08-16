from .contracts import AnalysisResult


class PhoenixPipeline:

    def __init__(self, model_hub):
        self.model_hub = model_hub

    def analyze(self, case):
        raw = self.model_hub.predict_all(case)

        result = AnalysisResult()

        result.raw_model_results = raw

        return result
