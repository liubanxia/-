import torch

from core.expert_result_fusion import ExpertFinding


class SegmentationResultBridge:

    def convert(self, results):
        findings = []

        for result in results:
            tensor = result.get("tensor")

            if not torch.is_tensor(tensor):
                continue

            x = tensor.detach().cpu()

            while x.ndim > 3:
                x = x[0]

            mask = x > 0

            coords = torch.nonzero(mask)

            if coords.numel() == 0:
                continue

            center = coords.float().mean(dim=0)

            if center.numel() < 3:
                continue

            z, y, xcoord = center[-3:].tolist()

            findings.append(
                ExpertFinding(
                    expert_id=result.get("expert_id", "segmentation"),
                    task="prompt_segmentation",
                    location="病灶区域",
                    finding="分割模型提示局灶异常区域",
                    impression="",
                    geometry={
                        "slice_index": int(round(z)),
                        "point": [float(xcoord), float(y)],
                        "tip": [float(xcoord), float(y)],
                    },
                    metadata={
                        "clinical_display": "pointer_only",
                    },
                )
            )

        return findings


SEGMENTATION_RESULT_BRIDGE = SegmentationResultBridge()
