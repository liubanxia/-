import shutil
from core.model_adapter import ModelAdapter


class TotalSegmentatorAdapter(ModelAdapter):
    name = "totalsegmentator"

    def load(self):
        if shutil.which("TotalSegmentator") is None:
            raise RuntimeError("TotalSegmentator CLI 尚未安装")

    def predict(self, case):
        return {"model": self.name, "status": "ready_for_anatomy_segmentation", "case_id": case.case_id}
