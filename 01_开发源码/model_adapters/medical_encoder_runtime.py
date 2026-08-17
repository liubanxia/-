from .hf_local_encoder import HFLocalEncoderAdapter


ENCODERS = {

    "medsiglip": HFLocalEncoderAdapter(
        "medsiglip",
        "D:/project_phoenix/04_AI模型/教师模型/13_MedSigLIP_448_ModelScope",
        "medical_vision_encoder",
    ),

    "rad_dino": HFLocalEncoderAdapter(
        "rad_dino",
        "D:/project_phoenix/04_AI模型/教师模型/15_RAD_DINO_MAIRA2_ModelScope",
        "radiology_vision_encoder",
    ),

    "m3d_clip": HFLocalEncoderAdapter(
        "m3d_clip",
        "D:/project_phoenix/04_AI模型/批量专家池/CT_通用/M3D-CLIP",
        "ct_3d_encoder",
    ),

    "biovil_t": HFLocalEncoderAdapter(
        "biovil_t",
        "D:/project_phoenix/04_AI模型/批量专家池/DR_胸片/BioViL-T",
        "cxr_text_encoder",
    ),
}


def get_encoder(model_id):
    return ENCODERS[model_id]


def list_encoders():
    return {
        name: {
            "task": adapter.task,
            "path": str(adapter.path),
            "status": adapter.status,
        }
        for name, adapter in ENCODERS.items()
    }
