from .case_router import XRAY_MODALITIES, MRI_MODALITIES, ct_route_decision, get_modalities, get_mri_region, get_xray_region, select_ct_specialists, select_initial_models, select_models
from .execution_status import execution_from_raw
from .lesion_geometry import resolve_lesions_for_case
from .model_pool_policy import MODEL_REGION_COVERAGE
from .model_roles import is_diagnostic_model, is_screening_model, model_role
from .result_fusion import fuse_results
from .report_generator import generate_report
from output.lesion_overlay import build_overlays


class PhoenixPipeline:
    """Portable two-stage medical-imaging inference pipeline."""

    def __init__(self, model_hub):
        self.model_hub = model_hub

    def _registered(self, names):
        registered = set(getattr(self.model_hub, "models", {}).keys())
        return [name for name in dict.fromkeys(names or []) if name in registered]

    def _execute_models(self, case, names):
        names = self._registered(names)
        if not names:
            return {}
        self.model_hub.load_selected(names)
        return self.model_hub.predict_selected(case, names)

    @staticmethod
    def _diagnostic_coverage(modalities, diagnostic_executions, screening_executions, routing_summary, helper_failures):
        selected = [item.model_name for item in diagnostic_executions]
        executed = [item.model_name for item in diagnostic_executions if item.executed and item.status == "success"]
        screening = [item.model_name for item in screening_executions if item.executed and item.status == "success"]
        gaps = []

        if "CT" in modalities:
            decision = routing_summary.get("ct_decision", {}) or {}
            routed = set(str(x) for x in decision.get("router_regions", []) or [])
            for region in ("head", "chest", "abdomen", "pelvis"):
                if decision.get(region):
                    routed.add(region)
            covered = set()
            for model_name in routing_summary.get("second_stage_models", []) or []:
                covered.update(MODEL_REGION_COVERAGE.get(model_name, set()))
            gaps = sorted(routed - covered)

        if selected:
            if len(executed) != len(selected):
                return {"status": "diagnostic_incomplete", "reason": "selected_diagnostic_model_not_completed", "regions_without_diagnostic_model": gaps, "failed_spatial_helpers": list(helper_failures)}
            if helper_failures:
                return {"status": "diagnostic_spatial_chain_incomplete", "reason": "localization_or_segmentation_failed", "regions_without_diagnostic_model": gaps, "failed_spatial_helpers": list(helper_failures)}
            if "CT" in modalities and gaps:
                return {"status": "diagnostic_partial_coverage", "reason": "some_routed_regions_have_no_validated_diagnostic_chain", "regions_without_diagnostic_model": gaps, "failed_spatial_helpers": []}
            return {"status": "diagnostic_complete", "reason": "selected_diagnostic_and_spatial_models_completed", "regions_without_diagnostic_model": [], "failed_spatial_helpers": []}

        if screening:
            return {"status": "screening_only", "reason": "screening_completed_but_no_diagnostic_model_selected", "regions_without_diagnostic_model": gaps, "failed_spatial_helpers": list(helper_failures)}
        if "CT" in modalities:
            return {"status": "router_only", "reason": "ct_anatomy_routed_but_no_validated_diagnostic_chain_selected", "regions_without_diagnostic_model": gaps, "failed_spatial_helpers": list(helper_failures)}
        if modalities & XRAY_MODALITIES:
            return {"status": "no_diagnostic_coverage", "reason": "xray_region_has_no_validated_diagnostic_chain", "regions_without_diagnostic_model": [str(routing_summary.get("xray_region", "other"))], "failed_spatial_helpers": list(helper_failures)}
        if modalities & MRI_MODALITIES:
            return {"status": "no_diagnostic_coverage", "reason": "mri_region_has_no_validated_diagnostic_chain", "regions_without_diagnostic_model": [str(routing_summary.get("mri_region", "other"))], "failed_spatial_helpers": list(helper_failures)}
        return {"status": "no_diagnostic_coverage", "reason": "no_diagnostic_model_selected", "regions_without_diagnostic_model": [], "failed_spatial_helpers": list(helper_failures)}

    def analyze(self, case):
        modalities = get_modalities(case)
        registered_models = list(getattr(self.model_hub, "models", {}).keys())
        raw = {}
        routing = {"mode": "single_stage", "initial_models": [], "unavailable_initial_models": [], "second_stage_candidates": [], "second_stage_models": [], "unavailable_second_stage_models": []}

        if "CT" in modalities:
            initial_candidates = select_initial_models(case)
            initial = self._registered(initial_candidates)
            routing["mode"] = "ct_two_stage"
            routing["initial_models"] = list(initial)
            routing["unavailable_initial_models"] = [x for x in initial_candidates if x not in initial]
            raw.update(self._execute_models(case, initial))
            router_result = raw.get("body_part_regression")
            candidates = select_ct_specialists(case, router_result)
            second = self._registered(candidates)
            routing["second_stage_candidates"] = list(candidates)
            routing["second_stage_models"] = list(second)
            routing["unavailable_second_stage_models"] = [x for x in candidates if x not in second]
            routing["ct_decision"] = ct_route_decision(case, router_result)
            raw.update(self._execute_models(case, second))
            selected = list(dict.fromkeys([*initial, *second]))
        else:
            if modalities & XRAY_MODALITIES:
                routing["xray_region"] = get_xray_region(case)
            if modalities & MRI_MODALITIES:
                routing["mri_region"] = get_mri_region(case)
            candidates = select_models(case)
            selected = self._registered(candidates)
            routing["initial_models"] = list(selected)
            routing["unavailable_initial_models"] = [x for x in candidates if x not in selected]
            raw.update(self._execute_models(case, selected))

        incomplete, helper_failures, executions = [], [], []
        for name in selected:
            execution = execution_from_raw(name, raw.get(name), self.model_hub.status.get(name, "not_loaded"), self.model_hub.errors.get(name, ""))
            executions.append(execution)
            if execution.status != "success":
                if model_role(name) in {"diagnostic", "router", "screening"}:
                    incomplete.append(name)
                else:
                    helper_failures.append(name)

        result = fuse_results(raw)
        resolved = resolve_lesions_for_case(case, result.lesions)
        diagnostic_executions = [x for x in executions if is_diagnostic_model(x.model_name)]
        screening_executions = [x for x in executions if is_screening_model(x.model_name)]
        coverage = self._diagnostic_coverage(modalities, diagnostic_executions, screening_executions, routing, helper_failures)
        hardware = getattr(self.model_hub, "hardware_profile", None)

        result.execution_summary = {
            "selected_models": list(selected),
            "models": [x.to_dict() for x in executions],
            "resolved_lesion_geometry": resolved,
            "critical_incomplete_models": list(incomplete),
            "helper_failed_models": list(helper_failures),
            "diagnostic_models_selected": [x.model_name for x in diagnostic_executions],
            "diagnostic_models_executed": [x.model_name for x in diagnostic_executions if x.executed and x.status == "success"],
            "screening_models_selected": [x.model_name for x in screening_executions],
            "screening_models_executed": [x.model_name for x in screening_executions if x.executed and x.status == "success"],
            "diagnostic_coverage": coverage,
            "routing": routing,
            "hardware_profile": hardware.to_dict() if hardware and hasattr(hardware, "to_dict") else {},
            "registered_models": registered_models,
        }
        result.diagnostic_executed = any(x.executed and x.status == "success" for x in diagnostic_executions)
        result.diagnostic_valid = bool(diagnostic_executions) and all(x.executed and x.status == "success" for x in diagnostic_executions) and not incomplete and not helper_failures and coverage.get("status") == "diagnostic_complete"
        result = generate_report(result, incomplete)
        return {
            "case_id": case.case_id,
            "selected_models": selected,
            "incomplete_models": incomplete,
            "helper_failed_models": helper_failures,
            "execution_summary": result.execution_summary,
            "diagnostic_executed": result.diagnostic_executed,
            "diagnostic_valid": result.diagnostic_valid,
            "analysis": result,
            "overlays": build_overlays(result.lesions),
        }
