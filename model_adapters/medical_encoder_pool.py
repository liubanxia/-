from .expert_base import PhoenixExpertAdapter


class M3DCLIPAdapter(PhoenixExpertAdapter):
    model_id = "m3d_clip"
    task = "ct_3d_encoder"

    def load(self):
        self.validate_assets()
        self.loaded = True
        return self

    def run(self, case):
        raise RuntimeError("M3D-CLIP forward pending unified debugging.")


class BioViLTAdapter(PhoenixExpertAdapter):
    model_id = "biovil_t"
    task = "cxr_encoder"

    def load(self):
        self.validate_assets()
        self.loaded = True
        return self

    def run(self, case):
        raise RuntimeError("BioViL-T forward pending unified debugging.")


class MedSigLIPAdapter(PhoenixExpertAdapter):
    model_id = "medsiglip"
    task = "medical_vision_encoder"

    def load(self):
        self.validate_assets()
        self.loaded = True
        return self

    def run(self, case):
        raise RuntimeError("MedSigLIP forward pending unified debugging.")
