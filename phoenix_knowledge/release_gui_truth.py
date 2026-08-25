from __future__ import annotations

import os
import time

_INSTALLED = False


def _deep_profile(profile: str) -> bool:
    return profile in {
        "deep",
        "4b",
        "deep4b",
        "quality",
        "max",
        "smart2",
    }


def install(gui_module) -> None:
    """Keep visible status/result labels aligned with actual execution."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow
    ask_worker_cls = gui_module.AskWorker
    original_refresh_translation_models = cls.refresh_translation_models

    def _status_text(self) -> str:
        try:
            status = self.workbench.status()
            semantic = str(
                status.get("semantic_label") or "语义状态未知"
            )
            smart2 = (
                "READY"
                if status.get("generator_deep_ready")
                else "未就绪"
            )
            # This is compute/provider readiness, not proof that local medical
            # translation stages M1/M2/M3 ran. Keep the label explicit so the
            # status bar cannot be mistaken for translation-route telemetry.
            text = (
                f"资料 {status['documents']} 本 | "
                f"知识块 {status['chunks']} | {semantic} | "
                f"算力Smart2={smart2}"
            )
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
            return text
        except Exception:
            return "Phoenix 状态读取失败"

    def refresh_translation_models(self):
        try:
            from .translation_chain_enforcement_v3 import chain_status

            engine = self.workbench.translator.engine
            chain = chain_status(engine)
            m1_names = chain.get("model1_names") or ()
            m1 = (
                "READY[" + ",".join(str(x) for x in m1_names) + "]"
                if chain.get("model1_ready")
                else "NOT READY"
            )
            m2 = "READY" if chain.get("model2_ready") else "NOT READY"
            m3 = "READY" if chain.get("model3_ready") else "NOT READY"
            api = "READY" if chain.get("api_ready") else "NOT READY"

            status = self.workbench.status()
            suffix = (
                " | 商业版已禁用非商业模型"
                if status.get("commercial_release")
                else ""
            )
            self.translation_models_label.setText(
                "正式医学翻译链："
                f"M1={m1} → M2={m2} → M3={m3} → "
                f"Smart2/API={api}（仅本地链失败时兜底）{suffix}"
            )
            tooltip = []
            if chain.get("model2_path"):
                tooltip.append("模型2：" + str(chain["model2_path"]))
            if chain.get("model3_path"):
                tooltip.append("模型3：" + str(chain["model3_path"]))
            self.translation_models_label.setToolTip("\n".join(tooltip))
        except Exception:
            return original_refresh_translation_models(self)

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
            requested = "Smart2"
            active_model = ""
            try:
                active_model = str(
                    self.workbench.llm.active_model_name(profile)
                )
            except Exception:
                active_model = ""

            if full.mode == "grounded_generation":
                label = requested
                if active_model:
                    if (
                        requested == "智能2"
                        and active_model == "Qwen3.5-2B"
                    ):
                        label = "智能2请求→Qwen3.5-2B降级执行"
                    else:
                        label = f"{requested} · {active_model}"
            elif full.mode == "grounding_blocked":
                label = "智能结果已被引用安全门拦截，已回退资料证据"
            elif deep_enabled:
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
    cls.refresh_translation_models = refresh_translation_models
    ask_worker_cls.run = run
