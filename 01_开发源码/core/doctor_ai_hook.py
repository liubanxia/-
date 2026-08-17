from core.clinical_case_controller import CLINICAL_CASE_CONTROLLER
from core.legacy_result_bridge import LEGACY_RESULT_BRIDGE


class DoctorAIHook:
    def __init__(self):
        self.doctor_triggered = False
        self.extended_started = False

    @staticmethod
    def _read(obj, *names):
        for name in names:
            value = getattr(obj, name, None)

            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue

            if value not in (None, ""):
                return value

        return None

    def case_from_window(self, window):
        service = self._read(
            window,
            "dicom_service",
            "_dicom_service",
            "inference_service",
            "_inference_service",
        )

        modality = self._read(
            window,
            "current_modality",
            "_current_modality",
        )

        body_part = self._read(
            window,
            "current_body_part",
            "_current_body_part",
        )

        path = self._read(
            window,
            "current_file",
            "_current_file",
            "current_dicom_path",
            "_current_dicom_path",
        )

        if service is not None:
            modality = modality or self._read(
                service,
                "modality",
                "current_modality",
            )

            path = path or self._read(
                service,
                "current_file",
                "current_path",
            )

        return {
            "modality": str(modality or ""),
            "body_part": str(body_part or ""),
            "study_description": "",
            "source_path": str(path or ""),
        }

    def on_doctor_click(self, window):
        self.doctor_triggered = True
        self.extended_started = False

        # 自动绑定 Phoenix 现有临床输出对象。
        try:
            from output.clinical_delivery_bridge import CLINICAL_DELIVERY_BRIDGE

            def first_attr(names):
                for name in names:
                    obj = getattr(window, name, None)
                    if obj is not None:
                        return obj
                return None

            report_writer = first_attr([
                "pacs_report_writer",
                "_pacs_report_writer",
                "report_writer",
                "_report_writer",
            ])

            lesion_memory = first_attr([
                "lesion_memory",
                "_lesion_memory",
            ])

            lesion_button = first_attr([
                "lesion_button",
                "_lesion_button",
            ])

            CLINICAL_DELIVERY_BRIDGE.bind(
                report_writer=report_writer,
                lesion_memory=lesion_memory,
                lesion_button=lesion_button,
            )
        except Exception:
            pass

        case_info = self.case_from_window(window)

        CLINICAL_CASE_CONTROLLER.open_case(
            case_info
        )

        return CLINICAL_CASE_CONTROLLER.doctor_start_ai()

    def accept_legacy_result(self, result):
        if not self.doctor_triggered:
            return None

        findings = LEGACY_RESULT_BRIDGE.convert(
            result
        )

        if not findings:
            return None

        bundle = CLINICAL_CASE_CONTROLLER.accept_expert_results(
            findings
        )

        # 原有CT/DR主链完成后，再串行启动扩展专家。
        if not self.extended_started:
            try:
                from core.expert_inference_scheduler import (
                    EXPERT_INFERENCE_SCHEDULER,
                )

                EXPERT_INFERENCE_SCHEDULER.start(
                    CLINICAL_CASE_CONTROLLER.case_info or {},
                    CLINICAL_CASE_CONTROLLER.execution_plan or {},
                )

                self.extended_started = True
            except Exception:
                pass

        return bundle

    def close_case(self):
        self.doctor_triggered = False
        self.extended_started = False

        try:
            from core.expert_inference_scheduler import (
                EXPERT_INFERENCE_SCHEDULER,
            )
            EXPERT_INFERENCE_SCHEDULER.stop()
        except Exception:
            pass

        try:
            from core.expert_feature_memory import (
                EXPERT_FEATURE_MEMORY,
            )
            EXPERT_FEATURE_MEMORY.clear()
        except Exception:
            pass

        CLINICAL_CASE_CONTROLLER.close_case()


DOCTOR_AI_HOOK = DoctorAIHook()
