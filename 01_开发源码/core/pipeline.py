from .case_router import (
    XRAY_MODALITIES,
    MRI_MODALITIES,
    ct_route_decision,
    get_modalities,
    get_mri_region,
    get_xray_region,
    select_ct_specialists,
    select_initial_models,
    select_models,
)
from .execution_status import execution_from_raw
from .lesion_geometry import resolve_lesions_for_case
from .model_pool_policy import MODEL_REGION_COVERAGE
from .model_roles import is_diagnostic_model, is_screening_model, model_role
from .result_fusion import fuse_results
from .report_generator import generate_report

from output.lesion_overlay import build_overlays


class PhoenixPipeline:

    def __init__(self, model_hub):
        self.model_hub = model_hub

    def _registered(self, names):
        registered = set(getattr(self.model_hub, "models", {}).keys())
        return [
            name
            for name in list(dict.fromkeys(names or []))
            if name in registered
        ]

    def _execute_models(self, case, names):
        names = self._registered(names)
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
        helper_failures,
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

        routed_regions = set()
        covered_regions = set()
        ct_gaps = []

        if "CT" in modalities:
            decision = routing_summary.get("ct_decision", {}) or {}
            routed_regions = set(
                str(region)
                for region in decision.get("router_regions", []) or []
            )
            for region in ("head", "chest", "abdomen", "pelvis"):
                if decision.get(region):
                    routed_regions.add(region)

            selected_specialists = set(
                routing_summary.get("second_stage_models", []) or []
            )
            for model_name in selected_specialists:
                covered_regions.update(
                    MODEL_REGION_COVERAGE.get(model_name, set())
                )

            ct_gaps = sorted(routed_regions - covered_regions)
            if not ct_gaps and not routed_regions and not diagnostic_selected:
                ct_gaps = ["undetermined_ct_region"]

        if diagnostic_selected:
            if len(diagnostic_executed) != len(diagnostic_selected):
                return {
                    "status": "diagnostic_incomplete",
                    "reason": "selected_diagnostic_model_not_completed",
                    "regions_without_diagnostic_model": ct_gaps,
                    "failed_spatial_helpers": list(helper_failures),
                }

            if helper_failures:
                return {
                    "status": "diagnostic_spatial_chain_incomplete",
                    "reason": "diagnostic_completed_but_localization_or_segmentation_failed",
                    "regions_without_diagnostic_model": ct_gaps,
                    "failed_spatial_helpers": list(helper_failures),
                }

            if "CT" in modalities and ct_gaps:
                return {
                    "status": "diagnostic_partial_coverage",
                    "reason": "some_routed_ct_regions_have_no_hospital_qualified_diagnostic_chain",
                    "regions_without_diagnostic_model": ct_gaps,
                    "failed_spatial_helpers": [],
                }

            return {
                "status": "diagnostic_complete",
                "reason": "selected_diagnostic_and_spatial_models_completed",
                "regions_without_diagnostic_model": [],
                "failed_spatial_helpers": [],
            }

        if screening_executed:
            return {
                "status": "screening_only",
                "reason": "screening_completed_but_no_diagnostic_model_selected",
                "regions_without_diagnostic_model": ct_gaps,
                "failed_spatial_helpers": list(helper_failures),
            }

        if "CT" in modalities:
            return {
                "status": "router_only",
                "reason": "ct_anatomy_routed_but_no_hospital_qualified_diagnostic_chain_selected",
                "regions_without_diagnostic_model": ct_gaps,
                "failed_spatial_helpers": list(helper_failures),
            }

        if modalities & XRAY_MODALITIES:
            region = str(routing_summary.get("xray_region", "other") or "other")
            return {
                "status": "no_diagnostic_coverage",
                "reason": "xray_region_has_no_hospital_qualified_diagnostic_chain",
                "regions_without_diagnostic_model": [region],
                "failed_spatial_helpers": list(helper_failures),
            }

        if modalities & MRI_MODALITIES:
            region = str(routing_summary.get("mri_region", "other") or "other")
            return {
                "status": "no_diagnostic_coverage",
                "reason": "mri_region_has_no_hospital_qualified_diagnostic_chain",
                "regions_without_diagnostic_model": [region],
                "failed_spatial_helpers": list(helper_failures),
            }

        return {
            "status": "no_diagnostic_coverage",
            "reason": "no_diagnostic_model_selected",
            "regions_without_diagnostic_model": [],
            "failed_spatial_helpers": list(helper_failures),
        }

    def analyze(self, case):
        modalities = get_modalities(case)
        registered_models = list(
            getattr(self.model_hub, "models", {}).keys()
        )
        raw = {}
        routing_summary = {
            "mode": "single_stage",
            "initial_models": [],
            "unavailable_initial_models": [],
            "second_stage_candidates": [],
            "second_stage_models": [],
            "unavailable_second_stage_models": [],
        }

        if "CT" in modalities:
            initial_candidates = select_initial_models(case)
            initial_models = self._registered(initial_candidates)
            routing_summary["mode"] = "ct_two_stage"
            routing_summary["initial_models"] = list(initial_models)
            routing_summary["unavailable_initial_models"] = [
                name for name in initial_candidates if name not in initial_models
            ]
            raw.update(self._execute_models(case, initial_models))

            router_result = raw.get("body_part_regression")
            second_stage_candidates = select_ct_specialists(
                case,
                router_result,
            )
            second_stage = self._registered(second_stage_candidates)
            routing_summary["second_stage_candidates"] = list(
                second_stage_candidates
            )
            routing_summary["second_stage_models"] = list(second_stage)
            routing_summary["unavailable_second_stage_models"] = [
                name
                for name in second_stage_candidates
                if name not in second_stage
            ]
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
            if modalities & XRAY_MODALITIES:
                routing_summary["xray_region"] = get_xray_region(case)
            if modalities & MRI_MODALITIES:
                routing_summary["mri_region"] = get_mri_region(case)

            initial_candidates = select_models(case)
            selected = self._registered(initial_candidates)
            routing_summary["initial_models"] = list(selected)
            routing_summary["unavailable_initial_models"] = [
                name for name in initial_candidates if name not in selected
            ]
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
                "定位/分割链未完整执行: "
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
            helper_failures,
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
            "model_pool_policy": getattr(
                self.model_hub,
                "model_pool_policy",
                {},
            ),
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
            and not incomplete
            and not helper_failures
            and diagnostic_coverage.get("status") == "diagnostic_complete"
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
