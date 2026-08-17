from pathlib import Path


MERLIN_ROOT = Path(
    "D:/project_phoenix/04_AI模型/批量专家池/CT_通用/Merlin"
)


MERLIN_COMPONENTS = {

    "merlin_ct_encoder": {
        "type": "vision_encoder",
        "checkpoint": MERLIN_ROOT /
        "nnUNetTrainerMerlin__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth",
        "prefix": "0.i3_resnet",
        "status": "REGISTERED_UNTESTED",
    },

    "merlin_clip_encoder": {
        "type": "vision_encoder",
        "checkpoint": MERLIN_ROOT /
        "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
        "prefix": "encode_image.i3_resnet",
        "status": "REGISTERED_UNTESTED",
    },

    "merlin_disease_backbone": {
        "type": "vision_encoder",
        "checkpoint": MERLIN_ROOT /
        "resnet_clinical_longformer_five_year_disease_prediction.pt",
        "prefix": "",
        "status": "REGISTERED_UNTESTED",
    },
}


def get_merlin_component(name):
    return MERLIN_COMPONENTS[name]
