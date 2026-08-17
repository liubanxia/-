from core.expert_execution_plan import (
    EXPERT_EXECUTION_PLAN,
)

from core.expert_runtime_bridge import (
    EXPERT_RUNTIME_BRIDGE,
)

from core.expert_result_fusion import (
    EXPERT_RESULT_FUSION,
)

from core.report_brain import (
    REPORT_BRAIN,
)


class PhoenixExpertStack:

    def prepare_case(
        self,
        modality,
        body_part=None,
        doctor_triggered=False,
    ):
        plan = EXPERT_EXECUTION_PLAN.build(
            modality=modality,
            body_part=body_part,
            doctor_triggered=doctor_triggered,
        )

        runtime = {
            group: EXPERT_RUNTIME_BRIDGE.resolve_many(
                plan.get(group, [])
            )
            for group in [
                "encoders",
                "segmentation",
                "report_teachers",
            ]
        }

        return {
            "plan": plan,
            "runtime": runtime,
        }

    def fuse(self, modality, expert_results):
        return EXPERT_RESULT_FUSION.fuse(
            modality,
            expert_results,
        )

    def build_report_context(
        self,
        case_info,
        fused_result,
    ):
        return REPORT_BRAIN.build_context(
            case_info,
            fused_result,
        )


PHOENIX_EXPERT_STACK = PhoenixExpertStack()
