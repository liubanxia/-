from pathlib import Path


def build_merlin_components(merlin_root):
    root = Path(merlin_root)
    return {
        "merlin_disease_backbone": {
            "type": "vision_encoder",
            "checkpoint": root / "resnet_clinical_longformer_five_year_disease_prediction.pt",
            "prefix": "",
            "status": "REGISTERED_UNTESTED",
            "storage_policy": "KEEP_SINGLE_CHECKPOINT",
        }
    }


MERLIN_COMPONENTS = {}
