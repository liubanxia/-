from .case_router import (
    ct_route_decision,
    get_modalities,
    select_ct_specialists,
    select_initial_models,
    select_models,
)
from .execution_status import execution_from_raw
from .lesion_geometry import resolve_lesions_for_case
from .model_roles import is_diagnostic_model, is_screening_model, model_role
from .result_fusion import fuse_results
from .report_generator import generate_report

from output.lesion_overlay import build_overlays


class PhoenixPipeline:

    def __init__(self, model_hub):
        self.model_hub = model_hub

    def _execute_models(self, case, names):
        names = list(dict.fromkeys(names or []))
        if not names:
            return {}

        self.model_hub.load_selected(names)
        return self.model_hub.predict_selected(case, names)

    @staticmethod
    def _diagnostic_coverage(
        modalities,
        diagnostic_executions,
        screening_executions,
        routing_summary,
    ):
        diagnostic_selected = [
            item.model_name for item in diagnostic_executions
        ]
        diagnostic_executed = [
            item.model_name
            for item in diagnostic_executions
            if item.executed and item.status == "success"
        ]
        screening_executed = [
            item.model_name
            for item in screening_executions
            if item.executed and item.status == "success"
        ]

        if diagnostic_selected:
            if len(diagnostic_executed) == len(diagnostic_selected):
                return {
                    "status": "diagnostic_complete",
                    "reason": "selected_diagnostic_models_completed",
                    "regions_without_diagnostic_model": [],
                }
            return {
                "status": "diagnostic_incomplete",
                "reason": "selected_diagnostic_model_not_completed",
                "regions_without_diagnostic_model": [],
            }

        if screening_executed:
            return {
                "status": "screening_only",
                "reason": "screening_completed_but_no_diagnostic_model_selected",
                "regions_without_diagnostic_model": [],
            }

        regions_without_model = []
        if "CT" in modalities:
            decision = routing_summary.get("ct_decision", {}) or {}
            routed_regions = decision.get("router_regions", []) or []
            # Current formal CT disease coverage is head + chest. A routed
            # abdomen/pelvis/other region with no specialist must be surfaced as
            # a coverage gap rather than looking like an AI-negative case.
            covered = {"head", "chest"}
            regions_without_model = [
                str(region)
                for region in routed_regions
                if str(region) not in covered
            ]
            return {
                "status": "router_only",
                "reason": "ct_anatomy_routed_but_no_diagnostic_model_selected",
                "regions_without_diagnostic_model": regions_without_model,
            }

        return {
            "status": "no_diagnostic_coverage",
            "reason": "no_diagnostic_or_screening_model_selected",
            "regions_without_diagnostic_model": [],
        }

    def analyze(self, case):
        modalities = get_modalities(case)
        raw = {}
        routing_summary = {
            "mode": "single_stage",
            "initial_models": [],
            "second_stage_models": [],
        }

        if "CT" in modalities:
            # Stage 1 must finish before specialist selection. This turns
            # BodyPartRegression from a passive audit model into the actual CT
            # anatomy router.
            initial_models = select_initial_models(case)
            routing_summary["mode"] = "ct_two_stage"
            routing_summary["initial_models"] = list(initial_models)
            raw.update(self._execute_models(case, initial_models))

            router_result = raw.get("body_part_regression")
            second_stage = select_ct_specialists(case, router_result)
            routing_summary["second_stage_models"] = list(second_stage)
            routing_summary["ct_decision"] = ct_route_decision(
                case,
                router_result,
            )

            raw.update(self._execute_models(case, second_stage))
            selected = list(dict.fromkeys([
                *initial_models,
                *second_stage,
            ]))
        else:
            selected = select_models(case)
            routing_summary["initial_models"] = list(selected)
            raw.update(self._execute_models(case, selected))

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

        registered_models = list(
            getattr(self.model_hub, "models", {}).keys()
        )
        registered_not_selected = [
            name
            for name in registered_models
            if name not in selected
        ]

        hardware = getattr(self.model_hub, "hardware_profile", None)
        if hardware is not None and hasattr(hardware, "to_dict"):
            hardware_summary = hardware.to_dict()
        else:
            hardware_summary = {}

        diagnostic_coverage = self._diagnostic_coverage(
            modalities,
            diagnostic_executions,
            screening_executions,
            routing_summary,
        )

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
            "diagnostic_coverage": diagnostic_coverage,
            "routing": routing_summary,
            "hardware_profile": hardware_summary,
            "registered_models": registered_models,
            "registered_models_not_selected": registered_not_selected,
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
