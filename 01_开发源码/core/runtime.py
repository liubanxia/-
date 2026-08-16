from core.case_session import CaseSession
from core.pipeline import PhoenixPipeline
from core.build_model_hub import build_full_model_hub

from pacs_io.factory import create_pacs_adapter
from output.lesion_memory import LesionMemory


class PhoenixRuntime:

    def __init__(self):
        self.session = CaseSession()
        self.memory = LesionMemory()

        self.adapter = None
        self.case = None

        self.model_hub = build_full_model_hub()
        self.pipeline = PhoenixPipeline(
            self.model_hub
        )

    def load_models(self):
        self.model_hub.load_all()

    def open_case(
        self,
        source,
        case_ref,
        **kwargs,
    ):
        self.close_case()

        self.session.open(
            str(case_ref)
        )

        self.adapter = create_pacs_adapter(
            source,
            **kwargs,
        )

        self.case = self.adapter.load_case(
            case_ref
        )

        return self.case

    def analyze(self):
        if self.case is None:
            raise RuntimeError("当前没有病例")

        return self.pipeline.analyze(
            self.case
        )

    def close_case(self):
        self.memory.clear()

        if self.adapter:
            try:
                self.adapter.close_case()
            except Exception:
                pass

        self.session.close()

        self.case = None
        self.adapter = None

    def shutdown(self):
        self.close_case()
        self.model_hub.unload_all()
