from output.result_window import ResultWindow
from output.pacs_report_writer import VendorPacsWriter
from output.lesion_button import LesionButton


class ResultDispatcher:

    def __init__(self, mode="A", pacs_writer=None):
        self.mode = mode.upper()
        self.pacs_writer = pacs_writer
        self.lesion_button = LesionButton()

    def show(self, case, result, memory):
        if self.mode == "A":
            ResultWindow().show(
                result,
                memory,
            )
            return {"status": "shown"}

        if self.mode == "B":
            writer = self.pacs_writer or VendorPacsWriter()

            write_result = writer.write_report(
                case,
                result["analysis"].report_draft,
            )

            if memory and memory.images:
                self.lesion_button.start(
                    memory
                )

            return write_result

        raise ValueError(
            f"未知输出模式: {self.mode}"
        )

    def close(self):
        self.lesion_button.close()


def dispatch_phoenix_clinical_result(
    case_info,
    expert_results,
    doctor_triggered=True,
):
    """
    Phoenix 新专家栈统一出口。
    不自动写PACS；只生成融合结果、报告上下文和病灶标记。
    """
    from core.clinical_case_controller import (
        CLINICAL_CASE_CONTROLLER,
    )

    controller = CLINICAL_CASE_CONTROLLER

    controller.open_case(case_info)

    if doctor_triggered:
        controller.doctor_start_ai()

    return controller.accept_expert_results(
        expert_results
    )


def dispatch_legacy_ai_result_to_phoenix(result):
    """
    已有 BodyPart / BLAST / TorchXRayVision /
    ResCBAM / FracAtlas 等结果统一进入 Phoenix 新临床栈。
    """
    from core.doctor_ai_hook import DOCTOR_AI_HOOK

    try:
        return DOCTOR_AI_HOOK.accept_legacy_result(
            result
        )
    except Exception:
        # 不允许新融合层破坏原有已工作的显示链
        return None
