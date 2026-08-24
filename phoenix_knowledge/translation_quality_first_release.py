from __future__ import annotations


_INSTALLED = False
_FINAL_TAG = "|quality_final_v2"


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import hymt_cascade_policy as hymt
    from . import translation_cascade_v2 as cascade
    from .qwen_local_medical_backend import LocalQwenMedicalBackend
    from .translation_models import TranslationAttempt

    # Formal medical translation is quality-first. Model1 is only a draft
    # candidate, HY-MT always gets a chance to correct/replace it, and model3
    # performs the final source-grounded medical edit. This deliberately avoids
    # the old early-exit thresholds where a merely plausible model1/model2
    # translation could become the published Office/PDF text.
    def quality_first_local_cascade(
        engine,
        source: str,
        target_language: str,
        attempts: list,
        errors: list[str],
    ):
        model1, _model1_passed = cascade._run_model1(
            engine,
            source,
            target_language,
            attempts,
            errors,
        )

        model2 = hymt._run_model2(
            engine,
            source,
            model1,
            target_language,
            attempts,
            errors,
        )

        base = model2 if model2 is not None and str(model2.text or "").strip() else model1
        if base is None or not str(base.text or "").strip():
            # Keep the source-only model3 recovery installed by
            # translation_local_first_release available for true model1/2
            # outages. Returning None here lets that wrapper handle it.
            return None, "quality_no_draft"

        if not cascade._model3_available(engine):
            return base, "quality_model3_unavailable"

        backend = cascade._model3(engine)
        try:
            text = backend.refine(source, base.text, target_language)
            attempt = hymt._quality_attempt(
                engine,
                backend.name + _FINAL_TAG,
                source,
                text,
                target_language,
            )
            attempts.append(attempt)
            return attempt, "quality_final_model3"
        except Exception as exc:
            errors.append(f"Qwen-model3-final: {type(exc).__name__}: {exc}")
            return base, "quality_model3_failed"

    cascade._run_local_cascade = quality_first_local_cascade

    previous_accept = cascade._local_draft_accepted

    def quality_first_accept(local_draft, local_stage: str) -> bool:
        if local_stage == "quality_final_model3":
            return bool(local_draft.quality.ok) and float(local_draft.quality.score) >= cascade.MODEL3_ACCEPT_SCORE
        if local_stage.startswith("quality_"):
            # Without a successful model3 final pass, keep the best local text
            # visible but mark it for review rather than silently publishing it
            # as fully accepted.
            return False
        return previous_accept(local_draft, local_stage)

    cascade._local_draft_accepted = quality_first_accept

    # Model3 is now a final editor, not merely a repair tool for text that has
    # already failed. The prompt therefore asks it to compare the two-stage
    # draft with the English source and change only what is necessary.
    def final_refine_prompt(self, source: str, draft: str, target_language: str) -> str:
        system = (
            "你是 Phoenix 本地医学文献终审翻译编辑。前面已有机器初译，现在必须逐句对照英文原文"
            "完成最终中文译文。医学准确性优先于文采。必须准确保持疾病、解剖、影像学、病理、"
            "检查技术、药物、统计学术语，以及所有数字、单位、正负号、侧别、否定关系、分级、"
            "诊断确定性、图表编号和医学缩写。缩写首次出现时可使用规范中文名并保留英文缩写。"
            "作者姓名、期刊名、DOI、URL、参考文献编号等引用信息不要擅自改写。标题和短标签应"
            "简洁准确翻译。禁止总结、删减、扩写、解释、拒答或添加原文没有的医学知识。"
            "只输出最终译文，不要说明过程。"
        )
        user = (
            f"目标语言：{target_language}\n\n"
            f"英文原文：\n{source}\n\n"
            f"两级初译：\n{draft}\n\n"
            "请逐句核对原文，修正术语、语义、否定、数字和遗漏，只输出最终完整译文。"
        )
        return self._chat_prompt(system, user)

    LocalQwenMedicalBackend._refine_prompt = final_refine_prompt

    # Invalidate stale Office unit checkpoints produced by the early-exit local
    # chain. Deterministic acronym/reference rows are left alone; only model1,
    # model2 and old untagged model3 translations are forced through the new
    # final-review contract once.
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
            if completed is None:
                return None
            _translated, _warnings, audits = completed
            for row in audits:
                if not isinstance(row, dict):
                    continue
                backend_name = str(row.get("backend", "") or "")
                local_old = (
                    backend_name.startswith("marian")
                    or backend_name.startswith("nllb")
                    or backend_name.startswith("hymt15_1p8b")
                    or (
                        backend_name.startswith("qwen_local_medical_model3")
                        and _FINAL_TAG not in backend_name
                    )
                )
                if local_old:
                    return None
            return completed

        OfficeDocumentTranslator._load_completed_unit = staticmethod(load_completed_unit)
    except Exception:
        pass

    print(
        "[Phoenix][翻译质量] 质量优先链已启用：模型1初稿→HY-MT模型2纠错→本地Qwen模型3逐段终审；API默认关闭。",
        flush=True,
    )
