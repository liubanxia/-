from model_adapters.ct_segmentation_runtime import (
    CT_SEGMENTATION_POOL,
)

from model_adapters.medical_encoder_runtime import (
    ENCODERS,
)

from model_adapters.vlm_component_pool import (
    VLM_COMPONENT_POOL,
)

from model_adapters.language_component_pool import (
    LANGUAGE_POOL,
)

from model_adapters.merlin_component_pool import (
    MERLIN_COMPONENTS,
)


class PhoenixExtendedPool:

    def __init__(self):
        self.ct_segmentation = CT_SEGMENTATION_POOL
        self.encoders = ENCODERS
        self.vlm_components = VLM_COMPONENT_POOL
        self.language_models = LANGUAGE_POOL
        self.merlin = MERLIN_COMPONENTS

    def summary(self):
        return {
            "ct_segmentation": len(self.ct_segmentation),
            "encoders": len(self.encoders),
            "vlm_components": len(self.vlm_components),
            "language_models": len(self.language_models),
            "merlin_components": len(self.merlin),
        }


EXTENDED_POOL = PhoenixExtendedPool()
