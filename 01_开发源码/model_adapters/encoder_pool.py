from pathlib import Path


class LazyEncoderAdapter:

    def __init__(self, model_id, path, task):
        self.model_id = model_id
        self.path = Path(path)
        self.task = task
        self.model = None
        self.status = "REGISTERED_UNTESTED"

    def validate_assets(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return True

    def unload(self):
        self.model = None


ENCODER_POOL = {
    "m3d_clip": LazyEncoderAdapter(
        "m3d_clip",
        "D:/project_phoenix/04_AI模型/批量专家池/CT_通用/M3D-CLIP",
        "ct_3d_encoder",
    ),

    "medsiglip": LazyEncoderAdapter(
        "medsiglip",
        "D:/project_phoenix/04_AI模型/教师模型/13_MedSigLIP_448_ModelScope",
        "medical_vision_encoder",
    ),

    "rad_dino": LazyEncoderAdapter(
        "rad_dino",
        "D:/project_phoenix/04_AI模型/教师模型/15_RAD_DINO_MAIRA2_ModelScope",
        "radiology_vision_encoder",
    ),

    "biovil_text": LazyEncoderAdapter(
        "biovil_text",
        "D:/project_phoenix/04_AI模型/批量专家池/DR_胸片/BioViL-T",
        "cxr_text_encoder",
    ),

    "merlin_ct": LazyEncoderAdapter(
        "merlin_ct",
        "D:/project_phoenix/04_AI模型/批量专家池/CT_通用/Merlin",
        "ct_3d_encoder",
    ),
}
