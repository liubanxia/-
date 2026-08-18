from .case_router import select_models
from .execution_status import execution_from_raw
from .lesion_geometry import resolve_lesions_for_case
from .model_roles import is_diagnostic_model, is_screening_model, model_role
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
        helper_failures = []
        executions = []

        for name in selected:
            load_status = self.model_hub.status.get(
                name,
                "not_loaded",
            )
            load_error = self.model_hub.errors.get(
                name,
                "",
            )

            execution = execution_from_raw(
                model_name=name,
                raw=raw.get(name),
                load_status=load_status,
                load_error=load_error,
            )
            executions.append(execution)

            if execution.status != "success":
                role = model_role(name)
                if role in {"diagnostic", "router", "screening"}:
                    incomplete.append(name)
                else:
                    helper_failures.append(name)

        result = fuse_results(raw)

        resolved_count = resolve_lesions_for_case(
            case,
            result.lesions,
        )

        if helper_failures:
            result.warnings.append(
                "辅助定位/分割模块未完整执行: "
                + ", ".join(helper_failures)
            )

        diagnostic_executions = [
            item
            for item in executions
            if is_diagnostic_model(item.model_name)
        ]

        screening_executions = [
            item
            for item in executions
            if is_screening_model(item.model_name)
        ]

        result.execution_summary = {
            "selected_models": list(selected),
            "models": [item.to_dict() for item in executions],
            "resolved_lesion_geometry": resolved_count,
            "critical_incomplete_models": list(incomplete),
            "helper_failed_models": list(helper_failures),
            "diagnostic_models_selected": [
                item.model_name
                for item in diagnostic_executions
            ],
            "diagnostic_models_executed": [
                item.model_name
                for item in diagnostic_executions
                if item.executed and item.status == "success"
            ],
            "screening_models_selected": [
                item.model_name
                for item in screening_executions
            ],
            "screening_models_executed": [
                item.model_name
                for item in screening_executions
                if item.executed and item.status == "success"
            ],
        }

        result.diagnostic_executed = any(
            item.executed and item.status == "success"
            for item in diagnostic_executions
        )

        result.diagnostic_valid = (
            bool(diagnostic_executions)
            and all(
                item.executed and item.status == "success"
                for item in diagnostic_executions
            )
        )

        result = generate_report(
            result,
            incomplete,
        )

        return {
            "case_id": case.case_id,
            "selected_models": selected,
            "incomplete_models": incomplete,
            "helper_failed_models": helper_failures,
            "execution_summary": result.execution_summary,
            "diagnostic_executed": result.diagnostic_executed,
            "diagnostic_valid": result.diagnostic_valid,
            "analysis": result,
            "overlays": build_overlays(
                result.lesions
            ),
        }
