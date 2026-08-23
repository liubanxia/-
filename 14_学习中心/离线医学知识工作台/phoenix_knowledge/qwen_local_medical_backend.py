from __future__ import annotations

from pathlib import Path


class LocalQwenMedicalBackend:
    """Phoenix local model3 backend.

    Runs the SSD-hosted Qwen medical refinement model before Smart2 API.
    """

    name = "qwen_local_medical_model3"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._tokenizer = None
        self._model = None
        self._device = None

    def available(self) -> bool:
        return self.model_path.exists() and any(self.model_path.iterdir())

    def _load(self):
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(f"Local Qwen model missing: {self.model_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            device_map="auto" if self._device == "cuda" else None,
        )
        self._model.eval()

    def refine(self, source: str, draft: str, target_language: str = "中文") -> str:
        self._load()

        prompt = f"""你是Phoenix医学翻译精修模型。
请根据英文原文检查现有中文译文。
保持医学含义、数字、单位、否定、分级和解剖关系准确。
不要总结，不要扩写。

英文原文:
{source}

现有译文:
{draft}

输出修正后的医学中文译文："""

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        import torch
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        return self._tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

    def unload(self):
        self._tokenizer = None
        self._model = None
