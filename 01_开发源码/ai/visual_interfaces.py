from abc import ABC, abstractmethod


class BaseVisualAI(ABC):
    """
    Project Phoenix 视觉AI统一接口。

    视觉A、视觉B以及未来替换的真实模型，
    都必须遵守此接口。
    """

    @property
    @abstractmethod
    def name(self):
        """返回视觉通路名称。"""
        raise NotImplementedError

    @abstractmethod
    def infer(self, series_context):
        """
        对已经通过Phoenix安全门控的影像Series执行推理。

        参数：
            series_context:
                后续由Phoenix统一定义的影像上下文。

        返回：
            当前M9.0-A阶段暂由具体实现返回测试结果。
            后续会统一为正式的候选病灶结果结构。
        """
        raise NotImplementedError


class VisualAInterface(BaseVisualAI):
    """
    视觉A：常规综合阅片AI。
    """

    @property
    def name(self):
        return "视觉A_综合阅片"

    @abstractmethod
    def infer(self, series_context):
        raise NotImplementedError


class VisualBInterface(BaseVisualAI):
    """
    视觉B：骨折漏诊防护AI。
    """

    @property
    def name(self):
        return "视觉B_骨折防护"

    @abstractmethod
    def infer(self, series_context):
        raise NotImplementedError
