from __future__ import annotations

import os


_INSTALLED = False
_API_FLAG = "PHOENIX_TRANSLATION_ALLOW_API_FALLBACK"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _api_fallback_allowed() -> bool:
    # Formal translation is offline/local-first by default. Remote API is only
    # an explicit emergency fallback, never a prerequisite for normal use.
    return _flag(_API_FLAG, default=False)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import hybrid_translation_policy as hybrid

    original_smart_available = hybrid._smart_available

    def translation_smart_available(engine) -> bool:
        if not _api_fallback_allowed():
            return False
        try:
            return bool(original_smart_available(engine))
        except Exception:
            return False

    # API is opt-in only. Every retained API fallback consults this gate.
    hybrid._smart_available = translation_smart_available

    # Restore the real first local translation stage. An older production
    # policy deliberately returned [] here because Smart2 had been the sole
    # formal translator. Once API became opt-in that old deny-list left the
    # local cascade with no model-1 input and could strand model3 as well.
    def local_model1_backends(engine, target_language: str) -> list[object]:
        target = str(target_language or "").strip().lower()
        if target not in {
            "中文",
            "简体中文",
            "chinese",
            "zh",
            "zh-cn",
        }:
            return []
        result: list[object] = []
        available = getattr(engine, "_backend_available", None)
        for attr in ("marian", "nllb"):
            backend = getattr(engine, attr, None)
            if backend is None:
                continue
            try:
                ready = bool(available(backend)) if callable(available) else bool(backend.available())
            except Exception:
                ready = False
            if ready:
                result.append(backend)
        if not bool(getattr(engine, "_phoenix_model1_inventory_reported", False)):
            names = ", ".join(
                str(getattr(item, "name", "local")) for item in result
            ) or "NOT READY"
            print(f"[Phoenix][模型1] {names}", flush=True)
            engine._phoenix_model1_inventory_reported = True
        return result

    hybrid._local_backends = local_model1_backends

    # Acronym disambiguation follows the same no-API-by-default policy.
    try:
        from .medical_acronyms import MedicalAcronymResolver

        original_llm_available = MedicalAcronymResolver._llm_available

        def acronym_llm_available(self) -> bool:
            try:
                if not original_llm_available(self):
                    return False
                llm = getattr(self, "llm", None)
                if llm is None:
                    return False
                backend = str(llm.backend("translation") or "")
                if backend == "remote_server" and not _api_fallback_allowed():
                    return False
                return True
            except Exception:
                return False

        MedicalAcronymResolver._llm_available = acronym_llm_available
    except Exception:
        pass

    # Install the real local chain in this order:
    # model1 Marian/NLLB -> model2 HY-MT -> model3 local Qwen -> optional API.
    from .hymt_cascade_policy import install as install_hymt
    from .translation_cascade_v2 import install as install_local_cascade

    install_hymt()
    install_local_cascade()

    # If both model1 and model2 are unavailable, model3 must still be able to
    # translate from source instead of requiring a previous Chinese draft.
    # This closes the dead-end that caused "1/2/3都不动工" after API was disabled.
    try:
        from . import translation_cascade_v2 as cascade_v2
        from . import hymt_cascade_policy as hymt

        previous_run_local_cascade = cascade_v2._run_local_cascade

        def run_local_cascade_with_model3_source_fallback(
            engine,
            source: str,
            target_language: str,
            attempts: list,
            errors: list[str],
        ):
            draft, stage = previous_run_local_cascade(
                engine,
                source,
                target_language,
                attempts,
                errors,
            )
            if draft is not None:
                return draft, stage
            if not cascade_v2._model3_available(engine):
                return None, stage

            backend = cascade_v2._model3(engine)
            try:
                system = (
                    "你是 Phoenix 本地医学翻译模型。把英文医学原文完整、准确地翻译成目标语言。"
                    "必须保留疾病、解剖、影像学、病理和检查技术术语，以及全部数字、单位、"
                    "正负号、侧别、否定关系、分级、医学缩写、图表编号和诊断确定性。"
                    "不得总结、删减、扩写或添加原文没有的医学知识。只输出完整译文。"
                )
                user = (
                    f"目标语言：{target_language}\n\n"
                    f"英文原文：\n{source}\n\n"
                    "请直接输出完整医学译文。"
                )
                backend._load()
                prompt = backend._chat_prompt(system, user)
                print(
                    "[Phoenix][模型3] 前两级无可用译文，启动本地源文直译。",
                    flush=True,
                )
                text = backend._generate_prompt(
                    prompt,
                    str(source or ""),
                    mode_label="源文直译",
                    max_input_length=1792,
                    max_output_tokens=768,
                )
                attempt = hymt._quality_attempt(
                    engine,
                    f"{backend.name}:source",
                    source,
                    text,
                    target_language,
                )
                attempts.append(attempt)
                return attempt, "model3_source"
            except Exception as exc:
                errors.append(f"Qwen-model3-source: {type(exc).__name__}: {exc}")
                return None, "model3_source_failed"

        cascade_v2._run_local_cascade = run_local_cascade_with_model3_source_fallback

        # Teach the acceptance gate about the model3 source-only stage.
        previous_local_draft_accepted = cascade_v2._local_draft_accepted

        def local_draft_accepted(local_draft, local_stage: str) -> bool:
            if local_stage == "model3_source":
                return bool(local_draft.quality.ok) and float(local_draft.quality.score) >= cascade_v2.MODEL3_ACCEPT_SCORE
            return previous_local_draft_accepted(local_draft, local_stage)

        cascade_v2._local_draft_accepted = local_draft_accepted
    except Exception as exc:
        print(
            f"[Phoenix][翻译] 模型3源文兜底安装失败: {type(exc).__name__}: {exc}",
            flush=True,
        )

    # Force Office/PPT batches through the local cascade instead of the old
    # Smart2 batch shortcut. That shortcut bypassed the API gate entirely and
    # also meant local models 1/2/3 could appear idle.
    try:
        from .translation_models import MultiModelTranslationEngine, _normalize_smart_level

        def local_first_translate_segments(
            self,
            sources: list[str],
            target_language: str = "中文",
            *,
            smart_level: str = "smart2",
        ):
            values = [str(value or "").strip() for value in sources]
            if not values:
                return ()
            level = _normalize_smart_level(smart_level)
            if level != "smart2":
                previous = getattr(self, "_phoenix_cascade_v2_previous_translate_segments")
                return previous(values, target_language, smart_level=level)
            return tuple(
                self.translate(value, target_language, smart_level="smart2")
                for value in values
            )

        MultiModelTranslationEngine.translate_segments = local_first_translate_segments
    except Exception as exc:
        print(
            f"[Phoenix][翻译] 本地批量路由安装失败: {type(exc).__name__}: {exc}",
            flush=True,
        )

    # Old warning-free checkpoints may have been produced by the previous
    # Smart2-qwen route while the compute gateway was in remote/API mode. Their
    # backend label does not distinguish local-vs-remote execution, so when API
    # fallback is disabled invalidate those legacy qwen35 unit caches once.
    try:
        from .office_translation import OfficeDocumentTranslator

        previous_load_completed = OfficeDocumentTranslator._load_completed_unit

        def load_completed_unit(
            path,
            unit,
            *,
            source_sha256: str,
            target_language: str,
            glossary_sha256: str,
        ):
            completed = previous_load_completed(
                path,
                unit,
                source_sha256=source_sha256,
                target_language=target_language,
                glossary_sha256=glossary_sha256,
            )
            if completed is None or _api_fallback_allowed():
                return completed
            _translated, _warnings, audits = completed
            for row in audits:
                if not isinstance(row, dict):
                    continue
                backend = str(row.get("backend", "") or "")
                if backend.startswith("qwen35_medical_translation"):
                    return None
            return completed

        OfficeDocumentTranslator._load_completed_unit = staticmethod(load_completed_unit)
    except Exception:
        pass

    print(
        "[Phoenix][翻译] 本地优先正式模式：模型1→HY-MT模型2→本地Qwen模型3；"
        "API默认关闭，仅设置 PHOENIX_TRANSLATION_ALLOW_API_FALLBACK=1 时最终兜底。",
        flush=True,
    )