from __future__ import annotations

import importlib.util
import ipaddress
import json
import os
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import WorkbenchPaths


VALID_MODES = {"auto", "cpu", "cuda", "deepspeed", "remote"}


@dataclass(frozen=True)
class ComputeSettings:
    mode: str = "auto"
    remote_url: str = ""
    remote_model_fast: str = ""
    remote_model_deep: str = ""


@dataclass(frozen=True)
class ComputeStatus:
    requested_mode: str
    effective_mode: str
    cuda_available: bool
    gpu_count: int
    gpu_names: tuple[str, ...]
    gpu_vram_gb: tuple[float, ...]
    deepspeed_available: bool
    remote_configured: bool
    remote_allowed: bool
    remote_url: str
    warning: str = ""

    def label(self) -> str:
        if self.effective_mode == "remote":
            host = urllib.parse.urlparse(self.remote_url).hostname or "外接服务"
            return f"外接GPU/API · {host}"
        if self.effective_mode == "deepspeed":
            names = ", ".join(self.gpu_names) or "CUDA GPU"
            return f"DeepSpeed · {names}"
        if self.effective_mode == "cuda":
            names = ", ".join(self.gpu_names) or "CUDA GPU"
            return f"本机GPU · {names}"
        return "CPU兼容"


class ComputeGateway:
    """Offline-first compute routing for the knowledge workbench.

    Modes:
    - auto: local CUDA when usable, otherwise CPU.
    - cpu: force local CPU.
    - cuda: prefer local CUDA, fall back to CPU.
    - deepspeed: use optional DeepSpeed inference on local CUDA; if DeepSpeed is
      unavailable or initialization fails, the caller can fall back to normal CUDA.
    - remote: explicitly authorized OpenAI-compatible GPU/API endpoint.

    Remote mode is never enabled by discovery. It requires both mode=remote and
    PHOENIX_KNOWLEDGE_ALLOW_REMOTE=1 in the current process. The API key is never
    written to disk by this class.
    """

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.config_path = paths.runtime_root / "compute_gateway.json"
        self._settings = self._load_settings()
        self._warning = ""

    @staticmethod
    def _flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_mode(value: str | None) -> str:
        raw = (value or "auto").strip().lower()
        aliases = {
            "gpu": "cuda",
            "local_gpu": "cuda",
            "ds": "deepspeed",
            "deep_speed": "deepspeed",
            "api": "remote",
            "external": "remote",
            "cloud": "remote",
        }
        raw = aliases.get(raw, raw)
        return raw if raw in VALID_MODES else "auto"

    def _load_settings(self) -> ComputeSettings:
        if not self.config_path.is_file():
            return ComputeSettings()
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return ComputeSettings()
        return ComputeSettings(
            mode=self._normalize_mode(payload.get("mode")),
            remote_url=str(payload.get("remote_url", "") or "").strip(),
            remote_model_fast=str(payload.get("remote_model_fast", "") or "").strip(),
            remote_model_deep=str(payload.get("remote_model_deep", "") or "").strip(),
        )

    def reload(self) -> None:
        self._settings = self._load_settings()
        self._warning = ""

    def save_settings(
        self,
        *,
        mode: str,
        remote_url: str = "",
        remote_model_fast: str = "",
        remote_model_deep: str = "",
    ) -> ComputeSettings:
        settings = ComputeSettings(
            mode=self._normalize_mode(mode),
            remote_url=str(remote_url or "").strip(),
            remote_model_fast=str(remote_model_fast or "").strip(),
            remote_model_deep=str(remote_model_deep or "").strip(),
        )
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.config_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.config_path)
        self._settings = settings
        return settings

    def requested_mode(self) -> str:
        env_mode = os.environ.get("PHOENIX_KNOWLEDGE_ACCELERATOR", "").strip()
        return self._normalize_mode(env_mode or self._settings.mode)

    def remote_url(self) -> str:
        return (
            os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_URL", "").strip()
            or self._settings.remote_url
        )

    def remote_api_key(self) -> str:
        return os.environ.get("PHOENIX_KNOWLEDGE_REMOTE_API_KEY", "").strip()

    def remote_allowed(self) -> bool:
        return self._flag("PHOENIX_KNOWLEDGE_ALLOW_REMOTE", default=False)

    @staticmethod
    def _cuda_info() -> tuple[bool, int, tuple[str, ...], tuple[float, ...]]:
        try:
            import torch

            if not torch.cuda.is_available():
                return False, 0, (), ()
            count = int(torch.cuda.device_count())
            names: list[str] = []
            memory: list[float] = []
            for index in range(count):
                try:
                    names.append(str(torch.cuda.get_device_name(index)))
                except Exception:
                    names.append(f"CUDA:{index}")
                try:
                    props = torch.cuda.get_device_properties(index)
                    memory.append(round(float(props.total_memory) / (1024 ** 3), 2))
                except Exception:
                    memory.append(0.0)
            return count > 0, count, tuple(names), tuple(memory)
        except Exception:
            return False, 0, (), ()

    @staticmethod
    def deepspeed_available() -> bool:
        try:
            return importlib.util.find_spec("deepspeed") is not None
        except Exception:
            return False

    def set_warning(self, text: str) -> None:
        self._warning = str(text or "").strip()

    def status(self) -> ComputeStatus:
        cuda, count, names, memory = self._cuda_info()
        ds = self.deepspeed_available()
        requested = self.requested_mode()
        remote_url = self.remote_url()
        remote_allowed = self.remote_allowed()
        warning = self._warning

        if requested == "remote":
            if remote_url and remote_allowed:
                effective = "remote"
            else:
                effective = "cpu"
                if not warning:
                    if not remote_url:
                        warning = "外接GPU/API未配置服务地址"
                    else:
                        warning = "外接GPU/API尚未获得本次会话授权"
        elif requested == "deepspeed":
            if cuda and ds:
                effective = "deepspeed"
            elif cuda:
                effective = "cuda"
                if not warning:
                    warning = "DeepSpeed不可用，已回退普通CUDA"
            else:
                effective = "cpu"
                if not warning:
                    warning = "未发现可用CUDA GPU，已回退CPU"
        elif requested == "cuda":
            effective = "cuda" if cuda else "cpu"
            if not cuda and not warning:
                warning = "未发现可用CUDA GPU，已回退CPU"
        elif requested == "cpu":
            effective = "cpu"
        else:
            effective = "cuda" if cuda else "cpu"

        return ComputeStatus(
            requested_mode=requested,
            effective_mode=effective,
            cuda_available=cuda,
            gpu_count=count,
            gpu_names=names,
            gpu_vram_gb=memory,
            deepspeed_available=ds,
            remote_configured=bool(remote_url),
            remote_allowed=remote_allowed,
            remote_url=remote_url,
            warning=warning,
        )

    def remote_model(self, profile: str | None = None) -> str:
        profile = (profile or "fast").strip().lower()
        deep = profile in {"deep", "4b", "deep4b", "quality", "max", "smart2"}
        env_name = (
            "PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP"
            if deep
            else "PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST"
        )
        configured = (
            os.environ.get(env_name, "").strip()
            or (
                self._settings.remote_model_deep
                if deep
                else self._settings.remote_model_fast
            )
        )
        if configured:
            return configured

        host = (urllib.parse.urlparse(self.remote_url()).hostname or "").lower()
        if host == "api.deepseek.com" or host.endswith(".deepseek.com"):
            return "deepseek-v4-pro" if deep else "deepseek-v4-flash"
        return "local-model"

    def remote_chat_url(self) -> str:
        raw = self.remote_url().strip().rstrip("/")
        if not raw:
            return ""
        parsed = urllib.parse.urlparse(raw)
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            return raw
        if path.endswith("/v1"):
            return raw + "/chat/completions"
        return raw + "/chat/completions"

    def remote_is_public(self) -> bool:
        host = (urllib.parse.urlparse(self.remote_url()).hostname or "").strip().lower()
        if not host:
            return False
        if host in {"localhost", "::1"}:
            return False
        try:
            ip = ipaddress.ip_address(host)
            return not (ip.is_loopback or ip.is_private or ip.is_link_local)
        except ValueError:
            if host.endswith(".local"):
                return False
            return True

    def is_deepseek_remote(self) -> bool:
        host = (urllib.parse.urlparse(self.remote_url()).hostname or "").lower()
        return host == "api.deepseek.com" or host.endswith(".deepseek.com")

    def local_device_map(self):
        mode = self.status().effective_mode
        if mode in {"cuda"}:
            return "auto"
        return None

    def wrap_deepspeed_inference(self, model):
        """Wrap one local CUDA model with DeepSpeed when explicitly requested.

        The desktop GUI defaults to tp_size=1. Advanced launchers can opt into
        a larger tensor-parallel degree with PHOENIX_KNOWLEDGE_DEEPSPEED_TP_SIZE;
        failures fall back to ordinary CUDA rather than breaking the workbench.
        """

        status = self.status()
        if status.effective_mode != "deepspeed":
            return model, None

        try:
            import deepspeed
            import torch

            local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
            requested_tp = max(1, int(os.environ.get("PHOENIX_KNOWLEDGE_DEEPSPEED_TP_SIZE", "1") or 1))
            available_gpus = max(1, int(torch.cuda.device_count()))
            world_size = min(requested_tp, available_gpus)
            torch.cuda.set_device(local_rank)
            model = model.to(torch.device(f"cuda:{local_rank}"))
            dtype = torch.float16
            engine = deepspeed.init_inference(
                model=model,
                config={
                    "dtype": dtype,
                    "tensor_parallel": {"tp_size": world_size},
                    "replace_with_kernel_inject": False,
                    "enable_cuda_graph": False,
                    "use_triton": False,
                },
            )
            module = getattr(engine, "module", model)
            self._warning = ""
            return module, engine
        except Exception as exc:
            self.set_warning(
                f"DeepSpeed初始化失败，已回退普通CUDA: {type(exc).__name__}: {exc}"
            )
            try:
                import torch

                model = model.to(torch.device("cuda:0"))
            except Exception:
                pass
            return model, None
