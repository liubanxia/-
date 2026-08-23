class DualVisionOrchestrator:
    """Run independent visual-A and visual-B paths behind a doctor-triggered gate."""

    def __init__(self, controller, visual_a, visual_b):
        self.controller = controller
        self.visual_a = visual_a
        self.visual_b = visual_b

    def infer(self, series_context):
        self.controller.assert_inference_allowed()

        result_a = None
        result_b = None
        error_a = None
        error_b = None

        try:
            result_a = self.visual_a.infer(series_context)
        except Exception as exc:
            error_a = str(exc)

        try:
            result_b = self.visual_b.infer(series_context)
        except Exception as exc:
            error_b = str(exc)

        if error_a is None and error_b is None:
            status = "success"
        elif error_a is not None and error_b is not None:
            status = "failed"
        else:
            status = "partial_failure"

        return {
            "status": status,
            "vision_a": {"result": result_a, "error": error_a},
            "vision_b": {"result": result_b, "error": error_b},
        }
