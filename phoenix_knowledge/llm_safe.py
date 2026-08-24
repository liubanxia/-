from __future__ import annotations

import json
import os
import socket
import urllib.request

from .llm import LocalLLM as _BaseLocalLLM


class LocalLLM(_BaseLocalLLM):
    """Product-safe LocalLLM with bounded HTTP waits.

    The original backend allowed a stalled local server to block for 10 minutes
    and a remote endpoint for 15 minutes. Product mode keeps the same protocol
    but bounds one request so pause/close can recover in a reasonable time.
    """

    @staticmethod
    def _timeout(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)) or default)
        except (TypeError, ValueError):
            value = default
        return max(15, min(value, 600))

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
        timeout = self._timeout("PHOENIX_KNOWLEDGE_LOCAL_TIMEOUT", 180)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, socket.error) as exc:
            raise RuntimeError(
                f"本地LLM服务在 {timeout}s 内未完成或不可用: {exc}"
            ) from exc

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
            raise RuntimeError(
                "公网外接GPU/API需要API密钥；密钥仅保存在当前进程环境变量中。"
            )

        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = self._timeout("PHOENIX_KNOWLEDGE_REMOTE_TIMEOUT", 180)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"外接GPU/API在 {timeout}s 内未完成或调用失败: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise RuntimeError(f"外接GPU/API响应格式异常: {data}") from exc
