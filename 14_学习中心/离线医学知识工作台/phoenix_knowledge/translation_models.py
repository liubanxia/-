from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import WorkbenchPaths, resolve_model_dir
from .llm import LocalLLM


_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%|mm|cm|mL|ml|mg|g|kg|HU|kV|mAs)?", re.I)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,10}\b")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _cuda_is_usable() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
        return int(major) >= 5
    except Exception:
        return False


def _normalize_number(token: str) -> str:
    return token.replace(",", "").lower()


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationAttempt:
    backend: str
    text: str
    quality: QualityReport
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationDecision:
    text: str
    backend: str
    quality: QualityReport
    needs_review: bool
    attempts: tuple[TranslationAttempt, ...] = field(default_factory=tuple)


class TranslationValidator:
    def validate(self, source: str, translated: str, target_language: str = "中文") -> QualityReport:
        source = (source or "").strip()
        translated = (translated or "").strip()
        reasons: list[str] = []
        score = 1.0

        if not translated:
            return QualityReport(False, 0.0, ("空译文",))

        if len(source) >= 80:
            ratio = len(translated) / max(len(source), 1)
            if ratio < 0.18:
                reasons.append("译文明显过短，疑似漏译")
                score -= 0.45
            elif ratio > 4.5:
                reasons.append("译文明显过长，疑似扩写或重复")
                score -= 0.3

        source_numbers = [_normalize_number(x) for x in _NUMBER_RE.findall(source)]
        if source_numbers:
            translated_numbers = {_normalize_number(x) for x in _NUMBER_RE.findall(translated)}
            kept = sum(1 for token in source_numbers if token in translated_numbers)
            coverage = kept / len(source_numbers)
            if coverage < 0.8:
                reasons.append(f"数字/单位保留率偏低({coverage:.0%})")
                score -= 0.35

        acronyms = list(dict.fromkeys(_ACRONYM_RE.findall(source)))
        if acronyms:
            kept = sum(1 for token in acronyms if token in translated)
            coverage = kept / len(acronyms)
            if coverage < 0.65:
                reasons.append(f"医学缩写保留率偏低({coverage:.0%})")
                score -= 0.2

        if target_language in {"中文", "简体中文", "Chinese", "zh", "zh-CN"}:
            cjk_count = len(_CJK_RE.findall(translated))
            latin_count = len(_LATIN_RE.findall(translated))
            source_latin = len(_LATIN_RE.findall(source))
            if source_latin >= 30 and cjk_count < 8:
                reasons.append("中文字符过少，疑似未翻译")
                score -= 0.55
            if source_latin >= 80 and latin_count > cjk_count * 7 and cjk_count < 30:
                reasons.append("英文残留比例过高")
                score -= 0.25

        if len(translated) >= 160:
            tail = translated[-80:]
            if tail and translated.count(tail) >= 3:
                reasons.append("检测到重复输出")
                score -= 0.25

        score = max(0.0, min(1.0, score))
        return QualityReport(score >= 0.62, score, tuple(reasons))


class _Seq2SeqBackend:
    name = "seq2seq"
    folder = ""

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.model_path = resolve_model_dir(paths.model_root, self.folder)
        self._tokenizer = None
        self._model = None
        self._device = None

    def available(self) -> bool:
        return self.model_path.exists() and any(self.model_path.iterdir())

    def _device_name(self) -> str:
        return "cuda:0" if _cuda_is_usable() else "cpu"

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class MarianEnZhBackend(_Seq2SeqBackend):
    name = "marian_en_zh"
    folder = "opus-mt-en-zh"

    def _load(self):
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(f"Marian翻译模型未下载: {self.model_path}")

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._device = self._device_name()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_path), local_files_only=True
        ).to(self._device)
        self._model.eval()
        if self._device == "cpu":
            self._model.to(dtype=torch.float32)

    def translate(self, text: str) -> str:
        import torch

        self._load()
        prefix = ""
        try:
            vocab = self._tokenizer.get_vocab()
            if ">>cmn_Hans<<" in vocab:
                prefix = ">>cmn_Hans<< "
            elif ">>zho_Hans<<" in vocab:
                prefix = ">>zho_Hans<< "
        except Exception:
            pass

        inputs = self._tokenizer(
            prefix + text,
            return_tensors="pt",
            truncation=True,
            max_length=480,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=700,
                renormalize_logits=True,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()


class NLLBEnZhBackend(_Seq2SeqBackend):
    name = "nllb_600m_en_zh"
    folder = "NLLB-200-distilled-600M"

    def _load(self):
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(f"NLLB翻译模型未下载: {self.model_path}")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._device = self._device_name()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            src_lang="eng_Latn",
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_path), local_files_only=True
        ).to(self._device)
        self._model.eval()

    def translate(self, text: str) -> str:
        import torch

        self._load()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=900,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        forced_bos = self._tokenizer.convert_tokens_to_ids("zho_Hans")
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                num_beams=4,
                max_new_tokens=900,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()


class QwenMedicalTranslationBackend:
    name = "qwen35_medical_review"

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def available(self) -> bool:
        return self.llm.available()

    def translate(self, text: str, target_language: str = "中文") -> str:
        prompt = f"""你是 Phoenix 医学书籍翻译器。把下面英文医学原文完整翻译成{target_language}。

要求：
- 不总结、不删减、不补充原文没有的医学知识。
- 保留所有数字、单位、影像学参数、分级、疾病名、药名、解剖部位、缩写、图表编号和参考文献编号。
- 医学术语优先使用规范中文；必要时首次出现可保留英文术语在括号中。
- 原文歧义或损坏时写“[原文不清]”，不得猜测。
- 只输出译文。

原文：
{text}
"""
        return self.llm.generate(prompt, max_new_tokens=1800).strip()


class MultiModelTranslationEngine:
    """Fail-safe offline cascade for whole-book translation.

    Order is intentionally translation-model first and general LLM last:
    Marian -> NLLB -> Qwen3.5. A result must pass structural checks before it
    stops the cascade. If every backend returns low-quality text, the best
    candidate is kept but explicitly marked for review by the caller.
    """

    def __init__(self, paths: WorkbenchPaths, llm: LocalLLM):
        self.paths = paths
        self.validator = TranslationValidator()
        self.marian = MarianEnZhBackend(paths)
        self.nllb = NLLBEnZhBackend(paths)
        self.qwen = QwenMedicalTranslationBackend(llm)

    def available_backends(self) -> list[str]:
        result = []
        for backend in (self.marian, self.nllb, self.qwen):
            if backend.available():
                result.append(backend.name)
        return result

    def translate(self, source: str, target_language: str = "中文") -> TranslationDecision:
        attempts: list[TranslationAttempt] = []
        best: TranslationAttempt | None = None
        backend_errors: list[str] = []

        for backend in (self.marian, self.nllb, self.qwen):
            if not backend.available():
                continue
            try:
                if isinstance(backend, QwenMedicalTranslationBackend):
                    text = backend.translate(source, target_language)
                else:
                    text = backend.translate(source)
                quality = self.validator.validate(source, text, target_language)
                attempt = TranslationAttempt(
                    backend=backend.name,
                    text=text,
                    quality=quality,
                )
                attempts.append(attempt)
                if best is None or attempt.quality.score > best.quality.score:
                    best = attempt
                if quality.ok:
                    return TranslationDecision(
                        text=text,
                        backend=backend.name,
                        quality=quality,
                        needs_review=False,
                        attempts=tuple(attempts),
                    )
            except Exception as exc:
                backend_errors.append(f"{backend.name}: {type(exc).__name__}: {exc}")

        if best is not None:
            return TranslationDecision(
                text=best.text,
                backend=best.backend,
                quality=best.quality,
                needs_review=True,
                attempts=tuple(attempts),
            )

        available = self.available_backends()
        if not available:
            raise RuntimeError(
                "没有可用翻译模型。至少下载 opus-mt-en-zh、NLLB-200-distilled-600M "
                "或 Qwen3.5-4B 中的一个。"
            )
        raise RuntimeError("所有翻译模型均执行失败: " + " | ".join(backend_errors))

    def unload(self) -> None:
        self.marian.unload()
        self.nllb.unload()
