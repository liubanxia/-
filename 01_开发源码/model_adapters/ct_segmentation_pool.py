from .expert_base import PhoenixExpertAdapter


class VISTA3DAdapter(PhoenixExpertAdapter):
    model_id = "vista3d"
    task = "ct_3d_segmentation"

    def load(self):
        self.validate_assets()

        # Unified lazy-load boundary.
        # Real framework/model construction is executed
        # only during final integration debugging.
        self.loaded = True
        return self

    def run(self, case):
        if not self.loaded:
            self.load()

        raise RuntimeError(
            "VISTA3D adapter registered; "
            "real forward is reserved for unified debugging."
        )


class SegVolAdapter(PhoenixExpertAdapter):
    model_id = "segvol"
    task = "ct_3d_segmentation"

    def load(self):
        self.validate_assets()
        self.loaded = True
        return self

    def run(self, case):
        if not self.loaded:
            self.load()

        raise RuntimeError(
            "SegVol adapter registered; "
            "real forward is reserved for unified debugging."
        )
