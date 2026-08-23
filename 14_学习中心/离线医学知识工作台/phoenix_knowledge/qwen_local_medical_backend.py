from __future__ import annotations

import time
from pathlib import Path


class LocalQwenMedicalBackend:
    """Phoenix local model3 backend.

    The same SSD-hosted Qwen2.5-3B instance supports two explicit modes:
    ``refine`` is the fallback translation repair stage; ``medical_review`` is
    the later full-page/unit medical editor.  The model stays resident until the
    translation task ends so page review does not repeatedly reload 3B weights.
    """

    name = "qwen_local_medical_model3"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._availability_reported = False
        self._first_refine_reported = False
        self._first_review_reported = False

    def available(self) -> bool:
        try:
            ready = (
                self.model_path.is_dir()
                and (self.model_path / "config.json").is_file()
                and any(
                    item.is_file()
                    and item.stat().st_size > 0
                    and item.suffix.lower() in {".safetensors", ".bin"}
                    for item in self.model_path.iterdir()
                )
            )
        except OSError:
            ready = False

        if not self._availability_reported:
            state = "READY" if ready else "NOT READY"
            print(
                f"[Phoenix][模型3] {state} | Qwen2.5-3B | {self.model_path}",
                flush=True,
            )
            self._availability_reported = True
        return ready

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(f"Local Qwen model missing/incomplete: {self.model_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(
            f"[Phoenix][模型3] 正在加载 Qwen2.5-3B -> {self._device}",
            flush=True,
        )
        if self._device == "cpu":
            print(
                "[Phoenix][模型3] 警告：当前 PyTorch 未检测到 CUDA，模型3将走CPU。",
                flush=True,
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )

            load_kwargs = {
                "local_files_only": True,
                "trust_remote_code": False,
                "low_cpu_mem_usage": True,
            }
            if self._device.startswith("cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["attn_implementation"] = "sdpa"

            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **load_kwargs,
                )
            except (TypeError, ValueError):
                load_kwargs.pop("attn_implementation", None)
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **load_kwargs,
                )

            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:
            print(
                f"[Phoenix][模型3] 加载失败: {type(exc).__name__}: {exc}",
                flush=True,
            )
            self.unload()
            raise

        gpu_name = ""
        if self._device.startswith("cuda"):
            try:
                gpu_name = f" | GPU={torch.cuda.get_device_name(0)} | FP16/SDPA"
            except Exception:
                gpu_name = ""
        print(
            f"[Phoenix][模型3] 已加载并启用 | device={self._device}{gpu_name}",
            flush=True,
        )

    def _chat_prompt(self, system: str, user: str) -> str:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"{system}\n\n{user}\n"

    def _refine_prompt(self, source: str, draft: str, target_language: str) -> str:
        system = (
            "你是 Phoenix 本地医学翻译修复器。当前译文来自前级翻译模型且未通过质量门槛。"
            "请严格核对英文原文并修正译文，不是总结。必须保持疾病、解剖、影像学、病理和"
            "检查技术术语准确；所有数字、单位、正负号、侧别、否定关系、分级、医学缩写、"
            "图表编号和诊断确定性必须与原文一致。不得删减、扩写或添加原文没有的医学知识。"
            "只输出完整修订译文。"
        )
        user = (
            f"目标语言：{target_language}\n\n"
            f"英文原文：\n{source}\n\n"
            f"前级译文：\n{draft}\n\n"
            "请修复错误并输出完整医学译文。"
        )
        return self._chat_prompt(system, user)

    def _review_prompt(self, source: str, draft: str, target_language: str) -> str:
        system = (
            "你是 Phoenix 医学文献终审编辑。翻译已经完成，现在执行整页/整单元复核。"
            "请逐项核对医学术语、解剖关系、疾病名称、影像学表现、病理/检查技术、数字、"
            "单位、正负号、侧别、否定关系、分级、缩写和图表编号。只改确实存在的问题；"
            "若原译文正确则保持原句。禁止总结、删减、扩写、改写诊断确定性或添加原文没有"
            "的信息。任何 <<<PHOENIX_SEGMENT_BOUNDARY>>> 标记必须原样保留且数量不变。"
            "只输出复核后的完整译文。"
        )
        user = (
            f"目标语言：{target_language}\n\n"
            f"英文原文/整页：\n{source}\n\n"
            f"当前完整译文：\n{draft}\n\n"
            "请完成医学复核，只输出最终译文。"
        )
        return self._chat_prompt(system, user)

    def _generate_prompt(
        self,
        prompt: str,
        draft: str,
        *,
        mode_label: str,
        max_input_length: int,
        max_output_tokens: int,
    ) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(1024, int(max_input_length)),
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        max_new_tokens = max(
            160,
            min(int(max_output_tokens), int(len(draft) * 1.15) + 128),
        )
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
        except torch.cuda.OutOfMemoryError as exc:
            print("[Phoenix][模型3] CUDA显存不足，已卸载模型3。", flush=True)
            self.unload()
            raise RuntimeError("Local Qwen model3 CUDA显存不足") from exc
        except Exception as exc:
            print(
                f"[Phoenix][模型3] {mode_label}失败: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        generated = output[0][input_length:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError(f"Local Qwen model3 returned empty text during {mode_label}")

        elapsed = max(time.perf_counter() - started, 0.001)
        token_count = int(generated.numel())
        print(
            f"[Phoenix][模型3] {mode_label}完成 | {elapsed:.1f}s | "
            f"{token_count / elapsed:.1f} token/s | {token_count} tokens",
            flush=True,
        )
        return text

    def refine(self, source: str, draft: str, target_language: str = "中文") -> str:
        source = str(source or "").strip()
        draft = str(draft or "").strip()
        if not source:
            raise ValueError("Local Qwen model3 requires a non-empty source")
        if not draft:
            raise ValueError("Local Qwen model3 is a refiner and requires a local draft")
        if not self._first_refine_reported:
            print("[Phoenix][模型3] 翻译级联触发：前两级失败，开始本地修复。", flush=True)
            self._first_refine_reported = True
        self._load()
        prompt = self._refine_prompt(source, draft, target_language)
        return self._generate_prompt(
            prompt,
            draft,
            mode_label="翻译修复",
            max_input_length=1792,
            max_output_tokens=768,
        )

    def medical_review(
        self,
        source: str,
        draft: str,
        target_language: str = "中文",
    ) -> str:
        source = str(source or "").strip()
        draft = str(draft or "").strip()
        if not source or not draft:
            raise ValueError("Local Qwen model3 medical review requires source and translated text")
        if not self._first_review_reported:
            print("[Phoenix][模型3] 进入第二阶段：整页/整单元医学复核。", flush=True)
            self._first_review_reported = True
        self._load()
        prompt = self._review_prompt(source, draft, target_language)
        return self._generate_prompt(
            prompt,
            draft,
            mode_label="整页医学复核",
            max_input_length=3072,
            max_output_tokens=1536,
        )

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
