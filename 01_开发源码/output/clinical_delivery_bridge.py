from core.clinical_output_pipeline import CLINICAL_OUTPUT_PIPELINE


class ClinicalDeliveryBridge:
    def __init__(self):
        self.report_writer = None
        self.lesion_memory = None
        self.lesion_button = None

    def bind(self, report_writer=None, lesion_memory=None, lesion_button=None):
        if report_writer is not None:
            self.report_writer = report_writer
        if lesion_memory is not None:
            self.lesion_memory = lesion_memory
        if lesion_button is not None:
            self.lesion_button = lesion_button
        return self

    @staticmethod
    def _call(obj, names, *args, **kwargs):
        if obj is None:
            return None

        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn(*args, **kwargs)

        return None

    def deliver_markers(self, markers):
        if self.lesion_memory is not None:
            self._call(
                self.lesion_memory,
                ("set_markers", "replace", "set_items", "set_results"),
                markers,
            )

        if self.lesion_button is not None and markers:
            self._call(
                self.lesion_button,
                ("show", "setVisible"),
                True,
            )

        return markers

    def write_report(self, report_text):
        return self._call(
            self.report_writer,
            ("write_report", "write", "set_report", "set_text"),
            report_text,
        )

    def prepare(self, case_info, fused_result, execution_plan):
        bundle = CLINICAL_OUTPUT_PIPELINE.prepare(
            case_info,
            fused_result,
            execution_plan,
        )

        self.deliver_markers(
            bundle["lesion_markers"]
        )

        return bundle

    def clear_case(self):
        self._call(
            self.lesion_memory,
            ("clear", "reset"),
        )

        if self.lesion_button is not None:
            self._call(
                self.lesion_button,
                ("hide", "close"),
            )


CLINICAL_DELIVERY_BRIDGE = ClinicalDeliveryBridge()
