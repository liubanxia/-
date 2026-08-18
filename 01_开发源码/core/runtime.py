from core.case_session import CaseSession
from core.pipeline import PhoenixPipeline
from core.build_model_hub import build_full_model_hub

from pacs_io.factory import create_pacs_adapter
from output.lesion_memory import LesionMemory


class PhoenixRuntime:

    def __init__(self):
        self.session = CaseSession()
        self.session.purge_stale()
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

        temp_dir = self.session.open(
            str(case_ref)
        )

        adapter = create_pacs_adapter(
            source,
            **kwargs,
        )

        try:
            case = adapter.load_case(
                case_ref
            )
        except Exception:
            try:
                adapter.close_case()
            except Exception:
                pass
            self.session.close()
            raise

        case.temp_dir = temp_dir

        self.adapter = adapter
        self.case = case

        return self.case

    def analyze(self):
        if self.case is None:
            raise RuntimeError("当前没有病例")

        result = self.pipeline.analyze(self.case)

        from output.lesion_capture import capture_lesions

        capture_lesions(
            self.case,
            result["analysis"].lesions,
            self.memory,
        )

        result["case_warnings"] = list(
            getattr(self.case, "warnings", []) or []
        )
        result["study_uid"] = str(
            getattr(self.case, "study_uid", "") or ""
        )
        result["source_path"] = str(
            getattr(self.case, "source_path", "") or ""
        )

        return result

    def close_case(self):
        try:
            from core.doctor_ai_hook import DOCTOR_AI_HOOK
            DOCTOR_AI_HOOK.close_case()
        except Exception:
            pass

        try:
            from core.clinical_case_controller import CLINICAL_CASE_CONTROLLER
            CLINICAL_CASE_CONTROLLER.close_case()
        except Exception:
            pass

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
