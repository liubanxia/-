from core.environment_paths import resolve_project_root


MERLIN_ROOT = (
    resolve_project_root()
    / "04_AI模型"
    / "批量专家池"
    / "CT_通用"
    / "Merlin"
)


MERLIN_COMPONENTS = {
    "merlin_disease_backbone": {
        "type": "vision_encoder",
        "checkpoint": (
            MERLIN_ROOT
            / "resnet_clinical_longformer_five_year_disease_prediction.pt"
        ),
        "prefix": "",
        "status": "REGISTERED_UNTESTED",
        "storage_policy": "KEEP_SINGLE_CHECKPOINT",
    },
}


def get_merlin_component(name):
    return MERLIN_COMPONENTS[name]
