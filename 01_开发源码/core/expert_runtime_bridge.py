from core.expert_router import EXPERT_ROUTER

from model_adapters.phoenix_extended_pool import (
    EXTENDED_POOL,
)

from model_adapters.native_specialist_runtime import (
    NATIVE_SPECIALISTS,
)


class ExpertRuntimeBridge:
    """
    把注册表里的逻辑专家名映射到实际 adapter。
    找不到运行实现时返回 None，不伪造模型结果。
    """

    def resolve(self, name):

        # CT 3D segmentation
        if name == "VISTA3D::segmentation_model":
            return EXTENDED_POOL.ct_segmentation.get(
                "vista3d"
            )

        if name == "SegVol::segmentation_model":
            return EXTENDED_POOL.ct_segmentation.get(
                "segvol"
            )

        # 医学 encoder
        encoder_map = {
            "M3D-CLIP::vision_encoder": "m3d_clip",
            "13_MedSigLIP_448_ModelScope::vision_encoder":
                "medsiglip",
            "RAD-DINO-MAIRA2::vision_encoder":
                "rad_dino",
            "BioViL-T::text_encoder":
                "biovil_t",
        }

        if name in encoder_map:
            return EXTENDED_POOL.encoders.get(
                encoder_map[name]
            )

        # 原生专科模型
        if name in NATIVE_SPECIALISTS:
            return NATIVE_SPECIALISTS[name]

        # VLM拆解组件
        if name in EXTENDED_POOL.vlm_components:
            return EXTENDED_POOL.vlm_components[name]

        if name in EXTENDED_POOL.language_models:
            return EXTENDED_POOL.language_models[name]

        # Merlin当前先保留组件引用
        if name.startswith("Merlin::"):
            return EXTENDED_POOL.merlin.get(
                name.split("::", 1)[1]
            )

        return None

    def resolve_many(self, names):
        out = {}

        for name in names:
            obj = self.resolve(name)

            if obj is not None:
                out[name] = obj

        return out


EXPERT_RUNTIME_BRIDGE = ExpertRuntimeBridge()
