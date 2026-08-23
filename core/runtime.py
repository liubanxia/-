from core.case_session import CaseSession
from core.pipeline import PhoenixPipeline
from core.build_model_hub import build_full_model_hub
from pacs_io.factory import create_pacs_adapter


class PhoenixRuntime:
    """Portable public runtime. No vendor-specific PACS write-back is included."""

    def __init__(self, model_root=None, temp_root=None):
        self.session = CaseSession(temp_root=temp_root)
        self.session.purge_stale()
        self.adapter = None
        self.case = None
        self.model_hub = build_full_model_hub(model_root=model_root)
        self.pipeline = PhoenixPipeline(self.model_hub)

    def open_case(self, source, case_ref, **kwargs):
        self.close_case()
        temp_dir = self.session.open(str(case_ref))
        adapter = create_pacs_adapter(source, **kwargs)
        try:
            case = adapter.load_case(case_ref)
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
        return case

    def analyze(self):
        if self.case is None:
            raise RuntimeError("当前没有病例")
        result = self.pipeline.analyze(self.case)
        result["case_warnings"] = list(getattr(self.case, "warnings", []) or [])
        result["study_uid"] = str(getattr(self.case, "study_uid", "") or "")
        return result

    def close_case(self):
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
