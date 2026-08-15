import numpy as np


class PacsAIBridge:
    def __init__(self, visual_b_manager):
        self.manager = visual_b_manager

    def run_dr(self, rgb):
        # PACS截图仅在RAM
        arr = np.asarray(rgb)

        if arr.ndim == 3:
            gray = (
                0.299 * arr[:,:,0]
                + 0.587 * arr[:,:,1]
                + 0.114 * arr[:,:,2]
            ).astype(np.uint8)
        else:
            gray = arr.astype(np.uint8)

        h, w = gray.shape

        ctx = {
            "modality": "DR",
            "modality_route": "DR",
            "current_image_array": gray,
            "image_array": gray,
            "rows": h,
            "columns": w,
            "source": "PACS_RAM_VIEWPORT",
        }

        result = self.manager.infer(ctx)

        # 只按真实结果字段判断，不靠字符串关键词
        def count_items(obj):
            det = mask = 0

            if isinstance(obj, dict):
                if "bbox_xyxy" in obj:
                    det += 1
                if "polygon_xy" in obj:
                    mask += 1

                for v in obj.values():
                    d, m = count_items(v)
                    det += d
                    mask += m

            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    d, m = count_items(v)
                    det += d
                    mask += m

            return det, mask

        detection_count, mask_count = count_items(result)
        positive = detection_count > 0 or mask_count > 0

        report = (
            "【检查类型】\nDR / X-ray\n\n"
            "【AI影像学所见】\n"
            + (
                f"视觉B发现骨折候选{detection_count}处，请医生核对标记区域。"
                if positive
                else
                "视觉B当前未发现明确超过阈值的骨折候选。"
            )
            + "\n\n【诊断意见】\n"
            "AI辅助结果，仅供医生结合原始PACS影像审核。"
        )

        return {
            "result": result,
            "detection_count": detection_count,
            "mask_count": mask_count,
            "report": report,
            "ram_only": True,
        }
