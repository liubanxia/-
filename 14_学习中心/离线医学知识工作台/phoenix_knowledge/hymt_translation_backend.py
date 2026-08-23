from __future__ import annotations

from pathlib import Path

from .config import WorkbenchPaths, model_dir_ready, resolve_model_dir


class HYMTMedicalTranslationBackend:
    """Second-stage local English->Chinese translation/refinement backend.

    The model is intentionally lazy-loaded because it is only used when the
    lightweight Marian/NLLB first pass does not meet Phoenix quality gates.
    """

    name = "hymt15_1p8b_refine"
    folder = "HY-MT1.5-1.8B"

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.model_path = resolve_model_dir(paths.model_root, self.folder)
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def available(self) -> bool:
        return model_dir_ready(self.model_path)

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

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.available():
            raise RuntimeError(f"HY-MT模型未下载完整: {self.model_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda:0" if self._cuda_is_usable() else "cpu"
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            torch_dtype=dtype,
        )
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _target_name(target_language: str) -> str:
        raw = str(target_language or "").strip().lower()
        if raw in {"中文", "简体中文", "chinese", "zh", "zh-cn"}:
            return "中文"
        return str(target_language or "中文")

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        self._load()
        import torch

        messages = [{"role": "user", "content": prompt}]
        tokenized = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        tokenized = tokenized.to(self._device)
        input_tokens = int(tokenized.shape[-1])
        with torch.inference_mode():
            output = self._model.generate(
                tokenized,
                max_new_tokens=max(256, min(2400, int(max_new_tokens))),
                do_sample=True,
                top_k=20,
                top_p=0.6,
                temperature=0.7,
                repetition_penalty=1.05,
            )
        generated = output[0][input_tokens:]
        return self._tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

    def translate(
        self,
        source: str,
        target_language: str = "中文",
    ) -> str:
        target = self._target_name(target_language)
        source = str(source or "").strip()
        prompt = (
            f"把下面英文医学文本完整、准确地翻译成{target}，只输出译文，不要解释。\n"
            "必须保留全部数字、单位、正负号、侧别、否定关系、分级、医学缩写、图表编号；"
            "不得总结、删减或扩写。\n\n"
            f"英文原文：\n{source}"
        )
        return self._generate(prompt, max_new_tokens=int(len(source) * 0.8) + 512)

    def refine(
        self,
        source: str,
        draft: str,
        target_language: str = "中文",
    ) -> str:
        target = self._target_name(target_language)
        source = str(source or "").strip()
        draft = str(draft or "").strip()
        context = (
            "下面是本地一级翻译模型生成的中文初稿。它可能存在医学术语、语序、漏译或生硬直译问题。\n"
            f"一级初稿：\n{draft}\n\n"
        )
        prompt = (
            context
            + f"参考上面的初稿，把下面英文原文重新核对并精修成{target}。"
            "只输出最终译文，不要解释；不得只润色初稿而忽略英文原文。"
            "必须逐句核对并保留数字、单位、正负号、侧别、否定关系、分级、医学缩写和图表编号，"
            "不得总结、删减或扩写。\n\n"
            f"英文原文：\n{source}"
        )
        return self._generate(prompt, max_new_tokens=int(len(source) * 0.8) + 512)

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def expected_model_path(paths: WorkbenchPaths) -> Path:
    return resolve_model_dir(paths.model_root, HYMTMedicalTranslationBackend.folder)
