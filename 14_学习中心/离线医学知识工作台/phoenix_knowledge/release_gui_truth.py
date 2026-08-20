from __future__ import annotations

import os
import time

_INSTALLED = False


def install(gui_module) -> None:
    """Keep visible status/result labels aligned with actual execution."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow
    ask_worker_cls = gui_module.AskWorker
    original_status_text = cls._status_text

    def _status_text(self) -> str:
        text = original_status_text(self)
        try:
            status = self.workbench.status()
            unresolved = int(
                status.get("document_paths_unresolved", 0) or 0
            )
            rebased = int(
                status.get("document_paths_rebased", 0) or 0
            )
            if unresolved:
                text += f" | ⚠资料路径待恢复={unresolved}"
            elif rebased:
                text += f" | SSD路径已自动重定位={rebased}"
        except Exception:
            pass
        return text

    def run(self):
        try:
            started = time.perf_counter()
            quick = self.workbench.ask(
                self.query,
                use_embeddings=False,
                deep=False,
            )
            quick_elapsed = time.perf_counter() - started
            self.completed.emit(
                f"【即时证据 {quick_elapsed:.2f}s】\n"
                f"{quick.text}\n\n"
                "—— 正在后台做语义补全；如已选择智能模式，将继续尝试带引用归纳 ——"
            )

            full_started = time.perf_counter()
            full = self.workbench.ask(self.query)
            full_elapsed = time.perf_counter() - full_started
            deep_enabled = os.environ.get(
                "PHOENIX_KNOWLEDGE_DEEP_QA",
                "0",
            ).strip().lower() in {"1", "true", "yes", "on"}
            profile = os.environ.get(
                "PHOENIX_KNOWLEDGE_LLM_PROFILE",
                "fast",
            ).strip().lower()

            if full.mode == "grounded_generation":
                requested = (
                    "智能2"
                    if profile
                    in {
                        "deep",
                        "4b",
                        "deep4b",
                        "quality",
                        "max",
                        "smart2",
                    }
                    else "智能1"
                )
                label = requested
            elif full.mode == "grounding_blocked":
                label = "智能结果已被引用安全门拦截，已回退资料证据"
            elif deep_enabled:
                requested = (
                    "智能2"
                    if profile
                    in {
                        "deep",
                        "4b",
                        "deep4b",
                        "quality",
                        "max",
                        "smart2",
                    }
                    else "智能1"
                )
                label = f"{requested}未实际生成，已回退资料证据"
            else:
                label = "快速证据"

            self.completed.emit(
                f"【完成 | {label} | 第二阶段 {full_elapsed:.2f}s】\n\n"
                f"{full.text}"
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    cls._status_text = _status_text
    ask_worker_cls.run = run
