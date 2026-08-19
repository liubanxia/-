from __future__ import annotations

import gc
import json
import os
import socket
import urllib.parse
import urllib.request

from .config import WorkbenchPaths, resolve_model_dir


class LocalLLM:
    """Offline-first generator with fast/deep model routing.

    Backends, in order:
    1) Explicit loopback OpenAI-compatible server.
    2) Local Qwen Transformers directory.
    3) No generator; caller falls back to evidence-only mode.

    Local model policy:
    - fast profile: prefer Qwen3.5-2B, fall back to Qwen3.5-4B.
    - deep profile: prefer Qwen3.5-4B, fall back to Qwen3.5-2B.

    Only one Transformers generator is kept resident at a time. Switching
    profiles unloads the previous model before loading the next one, preventing
    an 8 GB GPU from holding both models simultaneously.
    """

    model_name = "Qwen3.5-4B"  # compatibility alias for existing callers
    fast_model_name = "Qwen3.5-2B"
    deep_model_name = "Qwen3.5-4B"

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.fast_model_path = resolve_model_dir(
            paths.model_root,
            self.fast_model_name,
        )
        self.deep_model_path = resolve_model_dir(
            paths.model_root,
            self.deep_model_name,
        )
        # Preserve the old attribute for code/status that expects model_path.
        self.model_path = self.deep_model_path
        self.server_url = os.environ.get(
            "PHOENIX_KNOWLEDGE_LLM_URL", ""
        ).strip()
        self._processor = None
        self._model = None
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
        if raw in {"deep", "4b", "deep4b", "quality", "max"}:
            return "deep"
        return "fast"

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
        if self.server_url and self._is_loopback_url(self.server_url):
            return os.environ.get(
                "PHOENIX_KNOWLEDGE_LLM_MODEL", "local-server-model"
            )
        selected = self.selected_model(profile)
        return selected[0] if selected else "none"

    def backend(self, profile: str | None = None) -> str:
        if self.server_url:
            if not self._is_loopback_url(self.server_url):
                return "blocked_nonlocal_url"
            return "local_server"
        if self.selected_model(profile) is not None:
            return "transformers_local"
        return "evidence_only"

    def available(self, profile: str | None = None) -> bool:
        return self.backend(profile) in {"local_server", "transformers_local"}

    def _server_generate(self, prompt: str, max_new_tokens: int) -> str:
        if not self._is_loopback_url(self.server_url):
            raise RuntimeError(
                "知识工作台只允许连接本机回环地址，禁止外部API。"
            )

        payload = {
            "model": os.environ.get(
                "PHOENIX_KNOWLEDGE_LLM_MODEL", "local-model"
            ),
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

    def unload(self) -> None:
        """Release the resident generator before another heavy task/model."""

        self._processor = None
        self._model = None
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

        model_kwargs = {
            "local_files_only": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if self._cuda_is_usable():
            # Modern GPUs use automatic placement. Legacy hospital GPUs stay on
            # CPU because current PyTorch/CUDA builds do not support them well.
            model_kwargs["device_map"] = "auto"

        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_path),
            **model_kwargs,
        )
        model.eval()

        self._processor = processor
        self._model = model
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
        device = getattr(
            self._model,
            "device",
            next(self._model.parameters()).device,
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
            )
        generated = output[0][input_length:]
        return self._processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1200,
        *,
        profile: str | None = None,
    ) -> str:
        backend = self.backend(profile)
        if backend == "blocked_nonlocal_url":
            raise RuntimeError(
                "PHOENIX_KNOWLEDGE_LLM_URL 不是本机地址，已拒绝。"
            )
        if backend == "local_server":
            return self._server_generate(prompt, max_new_tokens)
        if backend == "transformers_local":
            return self._transformers_generate(
                prompt,
                max_new_tokens,
                profile=profile,
            )
        raise RuntimeError("未加载本地生成模型")
