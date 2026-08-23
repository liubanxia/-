from abc import ABC, abstractmethod


class BaseVisualAI(ABC):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self, series_context):
        raise NotImplementedError


class VisualAInterface(BaseVisualAI):
    @property
    def name(self):
        return "视觉A_综合阅片"

    @abstractmethod
    def infer(self, series_context):
        raise NotImplementedError


class VisualBInterface(BaseVisualAI):
    @property
    def name(self):
        return "视觉B_骨折防护"

    @abstractmethod
    def infer(self, series_context):
        raise NotImplementedError
