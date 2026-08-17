from core.report_orchestrator import REPORT_ORCHESTRATOR
from core.lesion_marker_bridge import LESION_MARKER_BRIDGE


class PhoenixClinicalOutputPipeline:

    def prepare(
        self,
        case_info,
        fused_result,
        execution_plan,
    ):
        markers = LESION_MARKER_BRIDGE.build(
            fused_result
        )

        prompt = REPORT_ORCHESTRATOR.build_prompt(
            case_info,
            fused_result,
        )

        return {
            "report_prompt": prompt,
            "lesion_markers": markers,
            "report_teachers": execution_plan.get(
                "report_teachers", []
            ),
        }

    def generate_report(
        self,
        case_info,
        fused_result,
        execution_plan,
    ):
        return REPORT_ORCHESTRATOR.generate(
            case_info,
            fused_result,
            allowed_teachers=execution_plan.get(
                "report_teachers"
            ),
        )


CLINICAL_OUTPUT_PIPELINE = PhoenixClinicalOutputPipeline()
