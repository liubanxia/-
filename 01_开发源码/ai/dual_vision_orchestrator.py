class DualVisionOrchestrator:
    """
    Project Phoenix 双视觉AI调度器。

    职责：
    1. 统一管理视觉A和视觉B的调用；
    2. 每次推理前强制检查医生主动启动门控；
    3. 视觉A、视觉B故障相互隔离；
    4. 当前M9.0-A阶段不负责真实模型加载；
    5. 不生成最终医学诊断。
    """

    def __init__(self, controller, visual_a, visual_b):
        self.controller = controller
        self.visual_a = visual_a
        self.visual_b = visual_b

    def infer(self, series_context):
        """
        执行一次双视觉推理。

        安全规则：
        - 未经医生主动启动时，两条视觉通路都不得运行；
        - 视觉A故障不得阻止视觉B运行；
        - 视觉B故障不得抹掉视觉A结果；
        - 任一通路失败时，不得把整体状态伪装成success。
        """

        # 所有视觉推理之前，先经过医生主动启动门控。
        self.controller.assert_inference_allowed()

        result_a = None
        result_b = None
        error_a = None
        error_b = None

        # --------------------------------------------------
        # 视觉A：综合阅片通路
        # --------------------------------------------------
        try:
            result_a = self.visual_a.infer(series_context)
        except Exception as exc:
            error_a = str(exc)

        # --------------------------------------------------
        # 视觉B：骨折漏诊防护通路
        # 即使视觉A失败，本通路仍必须独立执行。
        # --------------------------------------------------
        try:
            result_b = self.visual_b.infer(series_context)
        except Exception as exc:
            error_b = str(exc)

        # --------------------------------------------------
        # 汇总状态
        # --------------------------------------------------
        if error_a is None and error_b is None:
            status = "success"
        elif error_a is not None and error_b is not None:
            status = "failed"
        else:
            status = "partial_failure"

        return {
            "status": status,
            "vision_a": {
                "result": result_a,
                "error": error_a,
            },
            "vision_b": {
                "result": result_b,
                "error": error_b,
            },
        }
