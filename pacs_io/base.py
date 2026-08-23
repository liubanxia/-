from abc import ABC, abstractmethod
from .contracts import CaseInput


class PacsAdapter(ABC):

    @abstractmethod
    def load_case(self, case_ref: str) -> CaseInput:
        pass

    @abstractmethod
    def close_case(self) -> None:
        pass
