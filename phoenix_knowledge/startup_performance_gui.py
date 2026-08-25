from __future__ import annotations

import threading

from .compute_gateway import ComputeGateway, ComputeStatus


_INSTALLED = False
_CUDA_CACHE = None
_CUDA_LOCK = threading.RLock()


def _remote_status_without_cuda(gateway) -> ComputeStatus:
    remote_url = str(gateway.remote_url() or "").strip()
    remote_allowed = bool(gateway.remote_allowed())
    warning = str(getattr(gateway, "_warning", "") or "").strip()

    if remote_url and remote_allowed:
        effective = "remote"
    else:
        effective = "cpu"
        if not warning:
            warning = (
                "外接GPU/API未配置服务地址"
                if not remote_url
                else "外接GPU/API尚未获得本次会话授权"
            )

    if effective == "remote":
        try:
            if gateway.remote_is_public() and not gateway.remote_api_key():
                effective = "cpu"
                label = getattr(gateway, "provider_label", lambda: "云端模型")()
                warning = f"{label} API Key 未配置"
        except Exception:
            pass

    if effective == "remote":
        try:
            if not str(gateway.remote_model("smart2") or "").strip():
                effective = "cpu"
                label = getattr(gateway, "provider_label", lambda: "云端模型")()
                warning = f"{label} Smart2 模型 ID 未配置"
        except Exception:
            pass

    return ComputeStatus(
        requested_mode="remote",
        effective_mode=effective,
        cuda_available=False,
        gpu_count=0,
        gpu_names=(),
        gpu_vram_gb=(),
        deepspeed_available=False,
        remote_configured=bool(remote_url),
        remote_allowed=remote_allowed,
        remote_url=remote_url,
        warning=warning,
    )


def install(gui_module) -> None:
    """Make GUI startup cheap and defer hardware/model probing until use.

    The desktop used to call workbench.status()/translation readiness several
    times while constructing the window. Those paths enter torch.cuda probing,
    which is especially slow on older/unsupported CUDA installations. Remote
    API mode also had no reason to probe local CUDA at all.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    # Cache local CUDA discovery for the whole process. The first deliberate
    # local-hardware check may still cost time, but repeated status refreshes do not.
    original_cuda_info = ComputeGateway._cuda_info

    def cached_cuda_info():
        global _CUDA_CACHE
        with _CUDA_LOCK:
            if _CUDA_CACHE is None:
                try:
                    _CUDA_CACHE = original_cuda_info()
                except Exception:
                    _CUDA_CACHE = (False, 0, (), ())
            return _CUDA_CACHE

    ComputeGateway._cuda_info = staticmethod(cached_cuda_info)

    # Remote/API mode never needs torch.cuda discovery just to decide whether
    # the remote endpoint is ready. This removes the old-K10 startup penalty.
    original_compute_status = ComputeGateway.status

    def fast_compute_status(self):
        try:
            if str(self.requested_mode() or "").strip().lower() == "remote":
                return _remote_status_without_cuda(self)
        except Exception:
            pass
        return original_compute_status(self)

    ComputeGateway.status = fast_compute_status

    cls = gui_module.WorkbenchWindow
    original_init = cls.__init__
    original_refresh_translation_models = cls.refresh_translation_models
    original_status_text = cls._status_text
    original_update_compute_label = getattr(cls, "_update_compute_label", None)

    def refresh_translation_models(self):
        if bool(getattr(self, "_phoenix_fast_startup", False)):
            label = getattr(self, "translation_models_label", None)
            if label is not None:
                label.setText("医学精译：按需检测（开始翻译时自动检查 Smart2）")
            return None
        return original_refresh_translation_models(self)

    def status_text(self) -> str:
        if bool(getattr(self, "_phoenix_fast_startup", False)):
            return "Phoenix 已启动 | 资料状态已载入 | 模型/算力按需检测"
        # Keep ordinary status-bar refresh lightweight too. Detailed model and
        # compute truth belongs to the model/compute dialog and translation task.
        try:
            documents = self.workbench.db.list_documents()
            chunks = self.workbench.db.count_chunks()
            return f"资料 {len(documents)} 本 | 知识块 {chunks} | 模型/算力按需检测"
        except Exception:
            return original_status_text(self)

    def update_compute_label(self):
        if bool(getattr(self, "_phoenix_fast_startup", False)):
            label = getattr(self, "compute_status_label", None)
            if label is not None:
                label.setText("算力：按需检测")
                label.setToolTip("启动阶段不探测 CUDA；点击“模型/算力”时再检测。")
            return None
        if callable(original_update_compute_label):
            return original_update_compute_label(self)
        return None

    def fast_init(self, *args, **kwargs):
        self._phoenix_fast_startup = True
        try:
            original_init(self, *args, **kwargs)
        finally:
            self._phoenix_fast_startup = False
        try:
            self.statusBar().showMessage(
                "Phoenix 已启动；模型/算力将在实际使用时检测。",
                8000,
            )
        except Exception:
            pass

    cls.refresh_translation_models = refresh_translation_models
    cls._status_text = status_text
    if callable(original_update_compute_label):
        cls._update_compute_label = update_compute_label
    cls.__init__ = fast_init
    cls.__phoenix_fast_startup_contract__ = 1
    _INSTALLED = True
