from ai_models.expert_catalog import EXPERT_CATALOG


class PhoenixExpertRouter:

    CT_3D_ENCODERS = [
        "M3D-CLIP::vision_encoder",
        "Merlin::ct_3d_encoder",
        "Merlin::clip_ct_encoder",
    ]

    CT_SEGMENTATION = [
        "VISTA3D::segmentation_model",
        "SegVol::segmentation_model",
        "sam_med3d",
        "medsam2",
        "totalsegmentator",
    ]

    DR_ENCODERS = [
        "13_MedSigLIP_448_ModelScope::vision_encoder",
        "RAD-DINO-MAIRA2::vision_encoder",
    ]

    REPORT_TEACHERS = [
        "MedGemma-27B::language_model",
        "MedGemma-1.5-4B::language_model",
        "14_MAIRA_2_ModelScope::language_model",
        "Lingshu-32B::language_model",
        "Lingshu-7B::language_model",
        "HealthGPT-Pro-8B::language_model",
        "HealthGPT-Pro-4B::language_model",
        "Hulu-Med-4B::language_model",
        "Fleming-VL-8B::language_model",
    ]

    def __init__(self):
        self.catalog = EXPERT_CATALOG

    def resolve(self, group):
        groups = {
            "ct_encoder": self.CT_3D_ENCODERS,
            "ct_segmentation": self.CT_SEGMENTATION,
            "dr_encoder": self.DR_ENCODERS,
            "report_teacher": self.REPORT_TEACHERS,
        }

        names = groups.get(group, [])

        return [
            self.catalog.get(name)
            for name in names
            if self.catalog.get(name) is not None
        ]

    def monai_specialists(self):
        return {
            k: v
            for k, v in self.catalog.all().items()
            if k.startswith("monai::")
        }

    def suprem_specialists(self):
        return {
            k: v
            for k, v in self.catalog.all().items()
            if k.startswith("suprem::")
        }


EXPERT_ROUTER = PhoenixExpertRouter()
