from __future__ import annotations

import gc
import json
import os
import socket
import urllib.parse
import urllib.request

from .compute_gateway import ComputeGateway
from .config import WorkbenchPaths, resolve_model_dir


class LocalLLM:
    """Offline-first generator with local and explicitly authorized GPU routing.

    Backends, in order for normal local mode:
    1) Explicit loopback OpenAI-compatible server.
    2) Local Qwen Transformers directory.
    3) No generator; caller falls back to evidence-only mode.

    Optional compute modes:
    - auto/cpu/cuda: local execution.
    - deepspeed: optional DeepSpeed inference wrapper on local CUDA.
    - remote: explicitly authorized OpenAI-compatible external GPU/API service.

    Remote mode is never selected automatically and never stores an API key on disk.
    """

    model_name = "Qwen3.5-4B"
    fast_model_name = "Qwen3.5-2B"
    deep_model_name = "Qwen3.5-4B"

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.fast_model_path = resolve_model_dir(paths.model_root, self.fast_model_name)
        self.deep_model_path = resolve_model_dir(paths.model_root, self.deep_model_name)
        self.model_path = self.deep_model_path
        self.server_url = os.environ.get("PHOENIX_KNOWLEDGE_LLM_URL", "").strip()
        self.compute = ComputeGateway(paths)
        self._processor = None
        self._model = None
        self._model_engine = None
        self._loaded_model_name: str | None = None
        self._loaded_model_path = None

    @staticmethod
    def _is_loopback_url(url: str) -> bool:
        if not url:
            return False
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    @staticmethod
    def _cuda_is_usable() -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            major, _minor = torch.cuda.get_device_capability(0)
            return int(major) >= 5
        except Exception:
            return False

    @staticmethod
    def _path_ready(path) -> bool:
        try:
            return path.exists() and any(path.iterdir())
        except OSError:
            return False

    @staticmethod
    def _normalize_profile(profile: str | None) -> str:
        raw = (
            profile
            or os.environ.get("PHOENIX_KNOWLEDGE_LLM_PROFILE", "")
            or "fast"
        ).strip().lower()
        if raw in {"deep", "4b", "deep4b", "quality", "max", "smart2"}:
            return "deep"
        return "fast"

    def reload_compute_config(self) -> None:
        self.server_url = os.environ.get("PHOENIX_KNOWLEDGE_LLM_URL", "").strip()
        self.compute.reload()

    def compute_status(self) -> dict:
        status = self.compute.status()
        return {
            "requested_mode": status.requested_mode,
            "effective_mode": status.effective_mode,
            "label": status.label(),
            "cuda_available": status.cuda_available,
            "gpu_count": status.gpu_count,
            "gpu_names": list(status.gpu_names),
            "gpu_vram_gb": list(status.gpu_vram_gb),
            "deepspeed_available": status.deepspeed_available,
            "remote_configured": status.remote_configured,
            "remote_allowed": status.remote_allowed,
            "remote_url": status.remote_url,
            "warning": status.warning,
        }

    def _local_candidates(self, profile: str | None = None):
        normalized = self._normalize_profile(profile)
        if normalized == "deep":
            return (
                (self.deep_model_name, self.deep_model_path),
                (self.fast_model_name, self.fast_model_path),
            )
        return (
            (self.fast_model_name, self.fast_model_path),
            (self.deep_model_name, self.deep_model_path),
        )

    def selected_model(self, profile: str | None = None) -> tuple[str, object] | None:
        for name, path in self._local_candidates(profile):
            if self._path_ready(path):
                return name, path
        return None

    def active_model_name(self, profile: str | None = None) -> str:
        mode = self.compute.status().effective_mode
        if self.compute.requested_mode() == "remote":
            if mode == "remote":
                return self.compute.remote_model(self._normalize_profile(profile))
            return "remote-unavailable"
        if self.server_url and self._is_loopback_url(self.server_url):
            return os.environ.get("PHOENIX_KNOWLEDGE_LLM_MODEL", "local-server-model")
        selected = self.selected_model(profile)
        return selected[0] if selected else "none"

    def backend(self, profile: str | None = None) -> str:
        requested = self.compute.requested_mode()
        status = self.compute.status()

        if requested == "remote":
            return "remote_server" if status.effective_mode == "remote" else "remote_unavailable"

        if self.server_url:
            if not self._is_loopback_url(self.server_url):
                return "blocked_nonlocal_url"
            return "local_server"

        if self.selected_model(profile) is not None:
            return "transformers_local"
        return "evidence_only"

    def available(self, profile: str | None = None) -> bool:
        return self.backend(profile) in {
            "local_server",
            "remote_server",
            "transformers_local",
        }

    def _server_generate(self, prompt: str, max_new_tokens: int) -> str:
        if not self._is_loopback_url(self.server_url):
            raise RuntimeError("知识工作台本地服务只允许回环地址。")

        payload = {
            "model": os.environ.get("PHOENIX_KNOWLEDGE_LLM_MODEL", "local-model"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": int(max_new_tokens),
        }
        request = urllib.request.Request(
            self.server_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, socket.error) as exc:
            raise RuntimeError(f"本地LLM服务不可用: {exc}") from exc

        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise RuntimeError(f"本地LLM响应格式异常: {data}") from exc

    def _remote_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        profile: str | None = None,
    ) -> str:
        status = self.compute.status()
        if status.effective_mode != "remote":
            warning = status.warning or "外接GPU/API未授权或未配置"
            raise RuntimeError(warning)

        url = self.compute.remote_chat_url()
        if not url:
            raise RuntimeError("外接GPU/API服务地址为空")

        normalized_profile = self._normalize_profile(profile)
        payload: dict = {
            "model": self.compute.remote_model(normalized_profile),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": int(max_new_tokens),
            "stream": False,
        }
        if self.compute.is_deepseek_remote():
            payload["thinking"] = {
                "type": "enabled" if normalized_profile == "deep" else "disabled"
            }

        headers = {"Content-Type": "application/json"}
        api_key = self.compute.remote_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.compute.remote_is_public():
            raise RuntimeError("公网外接GPU/API需要API密钥；密钥仅保存在当前进程环境变量中。")

        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"外接GPU/API调用失败: {type(exc).__name__}: {exc}") from exc

        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise RuntimeError(f"外接GPU/API响应格式异常: {data}") from exc

    def unload(self) -> None:
        self._processor = None
        self._model = None
        self._model_engine = None
        self._loaded_model_name = None
        self._loaded_model_path = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_transformers(self, profile: str | None = None):
        selected = self.selected_model(profile)
        if selected is None:
            candidates = ", ".join(name for name, _ in self._local_candidates(profile))
            raise RuntimeError(f"本地生成模型未下载: {candidates}")

        model_name, model_path = selected
        if (
            self._model is not None
            and self._loaded_model_name == model_name
            and self._loaded_model_path == model_path
        ):
            return

        if self._model is not None:
            self.unload()

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        effective_mode = self.compute.status().effective_mode
        model_kwargs = {
            "local_files_only": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if effective_mode == "cuda" and self._cuda_is_usable():
            model_kwargs["device_map"] = "auto"

        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_path),
            **model_kwargs,
        )
        model.eval()

        engine = None
        if effective_mode == "deepspeed":
            model, engine = self.compute.wrap_deepspeed_inference(model)
        elif effective_mode == "cpu":
            try:
                model = model.to("cpu")
            except Exception:
                pass

        self._processor = processor
        self._model = model
        self._model_engine = engine
        self._loaded_model_name = model_name
        self._loaded_model_path = model_path

    def _transformers_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        profile: str | None = None,
    ) -> str:
        import torch

        self._load_transformers(profile)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            device = getattr(self._model, "device", None)
            if device is None:
                device = next(self._model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
            )
        generated = output[0][input_length:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1200,
        *,
        profile: str | None = None,
    ) -> str:
        backend = self.backend(profile)
        if backend == "blocked_nonlocal_url":
            raise RuntimeError("PHOENIX_KNOWLEDGE_LLM_URL 不是本机地址，已拒绝。")
        if backend == "remote_unavailable":
            status = self.compute.status()
            raise RuntimeError(status.warning or "外接GPU/API未授权或未配置")
        if backend == "remote_server":
            return self._remote_generate(prompt, max_new_tokens, profile=profile)
        if backend == "local_server":
            return self._server_generate(prompt, max_new_tokens)
        if backend == "transformers_local":
            return self._transformers_generate(
                prompt,
                max_new_tokens,
                profile=profile,
            )
        raise RuntimeError("未加载本地生成模型")
