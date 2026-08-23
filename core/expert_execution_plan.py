class ExpertExecutionPlan:
    def build(self, modality, body_part=None, doctor_triggered=False):
        modality = (modality or "").upper()
        body = (body_part or "").lower()
        plan = {
            "doctor_triggered": bool(doctor_triggered),
            "encoders": [],
            "segmentation": [],
            "lesion_models": [],
            "report_teachers": [],
        }
        if not doctor_triggered:
            return plan
        if modality == "CT":
            plan["encoders"] += ["M3D-CLIP::vision_encoder", "Merlin::ct_3d_encoder", "Merlin::clip_ct_encoder"]
            plan["segmentation"] += ["VISTA3D::segmentation_model", "SegVol::segmentation_model", "sam_med3d", "totalsegmentator"]
            if "head" in body or "brain" in body or "颅" in body:
                plan["lesion_models"].append("blast_ct")
        elif modality in {"DX", "DR", "CR", "XR"}:
            plan["encoders"] += ["13_MedSigLIP_448_ModelScope::vision_encoder", "RAD-DINO-MAIRA2::vision_encoder"]
            if any(x in body for x in ["chest", "lung", "thorax", "胸"]):
                plan["lesion_models"].append("torchxrayvision_chest")
            if any(x in body for x in ["bone", "skeletal", "骨"]):
                plan["lesion_models"] += ["yolov8_rescbam", "fracatlas_detect", "fracatlas_segment"]
        plan["report_teachers"] = [
            "MedGemma-1.5-4B::language_model", "14_MAIRA_2_ModelScope::language_model",
            "Hulu-Med-4B::language_model", "HealthGPT-Pro-4B::language_model",
        ]
        return plan


EXPERT_EXECUTION_PLAN = ExpertExecutionPlan()
