from core.phoenix_expert_stack import PHOENIX_EXPERT_STACK
from core.clinical_output_pipeline import CLINICAL_OUTPUT_PIPELINE
from output.clinical_delivery_bridge import CLINICAL_DELIVERY_BRIDGE


class ClinicalCaseController:
    def __init__(self):
        self.case_info = None
        self.execution_plan = None
        self.fused_result = None
        self.output_bundle = None
        self.report_draft = None

    def open_case(self, case_info):
        self.close_case()

        self.case_info = dict(case_info)

        prepared = PHOENIX_EXPERT_STACK.prepare_case(
            modality=case_info.get("modality"),
            body_part=case_info.get("body_part"),
            doctor_triggered=False,
        )

        self.execution_plan = prepared["plan"]
        return prepared

    def doctor_start_ai(self):
        if self.case_info is None:
            raise RuntimeError("No active case")

        prepared = PHOENIX_EXPERT_STACK.prepare_case(
            modality=self.case_info.get("modality"),
            body_part=self.case_info.get("body_part"),
            doctor_triggered=True,
        )

        self.execution_plan = prepared["plan"]
        return prepared

    def accept_expert_results(self, results):
        if self.case_info is None:
            raise RuntimeError("No active case")

        self.fused_result = PHOENIX_EXPERT_STACK.fuse(
            self.case_info.get("modality", ""),
            results,
        )

        try:
            from core.segmentation_prompt_memory import (
                SEGMENTATION_PROMPT_MEMORY,
            )

            SEGMENTATION_PROMPT_MEMORY.update_from_findings(
                self.fused_result.findings
            )
        except Exception:
            pass

        self.output_bundle = CLINICAL_DELIVERY_BRIDGE.prepare(
            self.case_info,
            self.fused_result,
            self.execution_plan or {},
        )

        # 先根据融合征象生成确定性的结构化报告初稿。
        try:
            from core.radiology_report_draft import (
                RADIOLOGY_REPORT_DRAFT,
            )

            self.report_draft = RADIOLOGY_REPORT_DRAFT.build(
                self.case_info,
                self.fused_result,
            )

            from output.ai_report_manager import (
                AI_REPORT_MANAGER,
            )

            if AI_REPORT_MANAGER.window is not None:
                AI_REPORT_MANAGER.window.set_report(
                    self.report_draft
                )
        except Exception:
            pass

        return self.output_bundle

    def generate_report(self):
        if self.fused_result is None:
            raise RuntimeError("No fused expert result")

        self.report_draft = CLINICAL_OUTPUT_PIPELINE.generate_report(
            self.case_info,
            self.fused_result,
            self.execution_plan or {},
        )

        try:
            from output.ai_report_manager import AI_REPORT_MANAGER

            if AI_REPORT_MANAGER.window is not None:
                AI_REPORT_MANAGER.window.set_report(
                    self.report_draft
                )
        except Exception:
            pass

        return self.report_draft

    def publish_draft(self, report_text):
        return CLINICAL_DELIVERY_BRIDGE.write_report(
            report_text
        )

    def show_ai_report_window(self, parent=None):
        from output.ai_report_manager import (
            AI_REPORT_MANAGER,
        )

        return AI_REPORT_MANAGER.show(
            controller=self,
            parent=parent,
        )

    def close_case(self):
        try:
            from output.ai_report_manager import AI_REPORT_MANAGER
            AI_REPORT_MANAGER.clear_case()
        except Exception:
            pass

        CLINICAL_DELIVERY_BRIDGE.clear_case()

        try:
            from core.segmentation_prompt_memory import (
                SEGMENTATION_PROMPT_MEMORY,
            )
            SEGMENTATION_PROMPT_MEMORY.clear()
        except Exception:
            pass

        self.case_info = None
        self.execution_plan = None
        self.fused_result = None
        self.output_bundle = None
        self.report_draft = None


CLINICAL_CASE_CONTROLLER = ClinicalCaseController()
