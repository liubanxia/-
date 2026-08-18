from __future__ import annotations

import json
import os
import socket
import urllib.parse
import urllib.request

from .config import WorkbenchPaths, resolve_model_dir


class LocalLLM:
    """Offline-first generator.

    Backends, in order:
    1) Explicit loopback OpenAI-compatible server.
    2) Local Qwen3.5-4B Transformers directory.
    3) No generator; caller falls back to evidence-only mode.
    """

    model_name = "Qwen3.5-4B"

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.model_path = resolve_model_dir(
            paths.model_root,
            self.model_name,
        )
        self.server_url = os.environ.get(
            "PHOENIX_KNOWLEDGE_LLM_URL", ""
        ).strip()
        self._processor = None
        self._model = None

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

    def backend(self) -> str:
        if self.server_url:
            if not self._is_loopback_url(self.server_url):
                return "blocked_nonlocal_url"
            return "local_server"
        if self.model_path.exists() and any(self.model_path.iterdir()):
            return "transformers_local"
        return "evidence_only"

    def available(self) -> bool:
        return self.backend() in {"local_server", "transformers_local"}

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

    def _load_transformers(self):
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise RuntimeError(f"本地生成模型未下载: {self.model_path}")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

        model_kwargs = {
            "local_files_only": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if self._cuda_is_usable():
            # Net-cafe/newer GPUs can use automatic device placement. Old
            # hospital GPUs (for example K10-class compute capability) are
            # intentionally kept on CPU instead of relying on unsupported CUDA.
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForMultimodalLM.from_pretrained(
            str(self.model_path),
            **model_kwargs,
        )
        self._model.eval()

    def _transformers_generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        import torch

        self._load_transformers()
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
            )
        generated = output[0][input_length:]
        return self._processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

    def generate(self, prompt: str, max_new_tokens: int = 1200) -> str:
        backend = self.backend()
        if backend == "blocked_nonlocal_url":
            raise RuntimeError(
                "PHOENIX_KNOWLEDGE_LLM_URL 不是本机地址，已拒绝。"
            )
        if backend == "local_server":
            return self._server_generate(prompt, max_new_tokens)
        if backend == "transformers_local":
            return self._transformers_generate(prompt, max_new_tokens)
        raise RuntimeError("未加载本地生成模型")
