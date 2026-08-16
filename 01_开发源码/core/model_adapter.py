from abc import ABC, abstractmethod


class ModelAdapter(ABC):

    name = "unknown"

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, case):
        pass

    def unload(self):
        pass
