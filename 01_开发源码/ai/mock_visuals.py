from ai.visual_interfaces import VisualAInterface, VisualBInterface


class MockVisualA(VisualAInterface):
    """
    M9.0-A 测试用视觉A。

    只用于验证双视觉调度流程，
    不执行真实医学影像推理。
    """

    def infer(self, series_context):
        return {
            "source": self.name,
            "status": "mock_success",
            "candidates": [],
        }


class MockVisualB(VisualBInterface):
    """
    M9.0-A 测试用视觉B。

    只用于验证骨折防护通路，
    不执行真实骨折检测。
    """

    def infer(self, series_context):
        return {
            "source": self.name,
            "status": "mock_success",
            "candidates": [],
        }
