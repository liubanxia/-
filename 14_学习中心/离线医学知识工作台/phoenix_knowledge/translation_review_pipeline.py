from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewStageResult:
    stage: str
    backend: str
    text: str
    changed: bool
    passed: bool
    accepted: bool
    quality_score: float
    reasons: tuple[str, ...] = ()
    error: str = ""


class MedicalTranslationReviewPipeline:
    """Second-pass medical review after the fallback translation cascade.

    The first phase chooses one usable translation through model1 -> model2 ->
    model3 -> Smart2 only when the previous stage fails.  This class is the
    second phase: the completed page/unit is reviewed in sequence by model1,
    model2, model3 and Smart2.  A later reviewer may improve the text, but a
    candidate that breaks the medical validator or segment structure is rejected
    and the last safe version is retained.
    """

    API_REVIEW_REASON = (
        "这是已经完成翻译并经过三级本地复核的医学译文。请做最终医学审稿，不要从头重译。"
        "只修正仍存在的医学术语、解剖关系、疾病名称、影像学表达、漏译/误译和中文语序问题；"
        "严格保持全部数字、单位、正负号、侧别、否定关系、分级、医学缩写、图表编号和诊断"
        "确定性。不得总结、删减、扩写或添加原文没有的信息。只输出最终完整译文。"
    )

    def __init__(self, engine):
        self.engine = engine

    def _quality(self, source: str, text: str, target_language: str):
        return self.engine.validator.validate(source, str(text or "").strip(), target_language)

    @staticmethod
    def _structure_ok(
        text: str,
        *,
        separator: str | None,
        expected_segments: int | None,
    ) -> bool:
        if not separator or not expected_segments or expected_segments <= 1:
            return True
        return str(text or "").count(separator) == int(expected_segments) - 1

    def _accept_candidate(
        self,
        source: str,
        current: str,
        candidate: str,
        target_language: str,
        *,
        separator: str | None,
        expected_segments: int | None,
    ) -> tuple[str, bool, bool, float, tuple[str, ...]]:
        candidate = str(candidate or "").strip()
        current = str(current or "").strip()
        if not candidate:
            current_quality = self._quality(source, current, target_language)
            return current, False, False, float(current_quality.score), ("empty_candidate",)
        if not self._structure_ok(
            candidate,
            separator=separator,
            expected_segments=expected_segments,
        ):
            current_quality = self._quality(source, current, target_language)
            return current, False, False, float(current_quality.score), ("segment_structure_changed",)

        current_quality = self._quality(source, current, target_language)
        candidate_quality = self._quality(source, candidate, target_language)
        passed = bool(candidate_quality.ok)
        accepted = bool(
            candidate_quality.ok
            and (
                not current_quality.ok
                or float(candidate_quality.score) >= float(current_quality.score)
            )
        )
        if accepted:
            return (
                candidate,
                candidate != current,
                True,
                float(candidate_quality.score),
                tuple(candidate_quality.reasons),
            )
        return (
            current,
            False,
            passed,
            float(candidate_quality.score),
            tuple(candidate_quality.reasons),
        )

    def _model1_candidate(self, source: str, target_language: str) -> tuple[str | None, str, str]:
        try:
            from . import hybrid_translation_policy as hybrid

            best = None
            for backend in hybrid._local_backends(self.engine, target_language):
                try:
                    attempt = hybrid._attempt(self.engine, backend, source, target_language)
                except Exception:
                    continue
                if best is None or float(attempt.quality.score) > float(best.quality.score):
                    best = attempt
            if best is None:
                return None, "model1", "unavailable"
            return best.text, best.backend, ""
        except Exception as exc:
            return None, "model1", f"{type(exc).__name__}: {exc}"

    def _model2_candidate(
        self,
        source: str,
        current: str,
        target_language: str,
    ) -> tuple[str | None, str, str]:
        try:
            from . import hymt_cascade_policy as cascade

            if not cascade._model2_available(self.engine):
                return None, "hymt15_1p8b", "unavailable"
            backend = cascade._model2(self.engine)
            return backend.refine(source, current, target_language), backend.name, ""
        except Exception as exc:
            return None, "hymt15_1p8b", f"{type(exc).__name__}: {exc}"

    def _model3_candidate(
        self,
        source: str,
        current: str,
        target_language: str,
    ) -> tuple[str | None, str, str]:
        try:
            from . import translation_cascade_v2 as cascade_v2

            if not cascade_v2._model3_available(self.engine):
                return None, "qwen_local_medical_model3", "unavailable"
            backend = cascade_v2._model3(self.engine)
            review = getattr(backend, "medical_review", None)
            if callable(review):
                text = review(source, current, target_language)
            else:
                text = backend.refine(source, current, target_language)
            return text, backend.name, ""
        except Exception as exc:
            return None, "qwen_local_medical_model3", f"{type(exc).__name__}: {exc}"

    def _api_candidate(
        self,
        source: str,
        current: str,
        target_language: str,
    ) -> tuple[str | None, str, str]:
        try:
            from . import hybrid_translation_policy as hybrid

            if not hybrid._smart_available(self.engine):
                return None, "smart2_api", "unavailable"
            quality = self._quality(source, current, target_language)
            reasons = tuple((*quality.reasons, self.API_REVIEW_REASON))
            text = self.engine.qwen.retry_translation(
                source,
                current,
                reasons,
                target_language,
            )
            return text, f"{self.engine.qwen.name}_final_page_review", ""
        except Exception as exc:
            return None, "smart2_api", f"{type(exc).__name__}: {exc}"

    def run(
        self,
        source_text: str,
        translated_text: str,
        target_language: str = "中文",
        *,
        separator: str | None = None,
        expected_segments: int | None = None,
        label: str = "整页",
    ) -> tuple[str, tuple[ReviewStageResult, ...]]:
        source = str(source_text or "").strip()
        current = str(translated_text or "").strip()
        if not source or not current:
            return current, ()

        print(f"[Phoenix][复核] {label} 开始：模型1 -> 模型2 -> 模型3 -> Smart2", flush=True)
        results: list[ReviewStageResult] = []
        stages = (
            ("model1_review", lambda value: self._model1_candidate(source, target_language)),
            ("model2_review", lambda value: self._model2_candidate(source, value, target_language)),
            ("model3_review", lambda value: self._model3_candidate(source, value, target_language)),
            ("api_review", lambda value: self._api_candidate(source, value, target_language)),
        )

        for stage, runner in stages:
            before = current
            candidate, backend, error = runner(current)
            if candidate is None:
                quality = self._quality(source, current, target_language)
                result = ReviewStageResult(
                    stage=stage,
                    backend=backend,
                    text=current,
                    changed=False,
                    passed=bool(quality.ok),
                    accepted=False,
                    quality_score=float(quality.score),
                    reasons=tuple(quality.reasons),
                    error=error,
                )
            else:
                current, changed, passed, score, reasons = self._accept_candidate(
                    source,
                    current,
                    candidate,
                    target_language,
                    separator=separator,
                    expected_segments=expected_segments,
                )
                result = ReviewStageResult(
                    stage=stage,
                    backend=backend,
                    text=current,
                    changed=changed,
                    passed=passed,
                    accepted=current != before,
                    quality_score=score,
                    reasons=reasons,
                    error=error,
                )
            results.append(result)
            suffix = f" | error={result.error}" if result.error and result.error != "unavailable" else ""
            state = "CHANGED" if result.changed else ("PASS" if result.passed else "REJECT")
            print(
                f"[Phoenix][复核] {stage} | {backend} | {state} | score={result.quality_score:.2f}{suffix}",
                flush=True,
            )

        final_quality = self._quality(source, current, target_language)
        print(
            f"[Phoenix][复核] {label} 完成 | final={'PASS' if final_quality.ok else 'REVIEW'} "
            f"| score={float(final_quality.score):.2f}",
            flush=True,
        )
        return current, tuple(results)
