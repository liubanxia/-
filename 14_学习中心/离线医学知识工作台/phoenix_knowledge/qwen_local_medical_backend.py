from __future__ import annotations

import time
from pathlib import Path


class LocalQwenMedicalBackend:
    """Phoenix local model3 backend.

    Runs the SSD-hosted Qwen2.5-3B refinement model before Smart2 API.
    The backend is fully offline and returns only newly generated answer tokens,
    never the prompt itself.
    """

    name = "qwen_local_medical_model3"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._availability_reported = False
        self._first_refine_reported = False

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
                # Older transformers builds may not expose SDPA as a load arg.
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

    def _prompt(self, source: str, draft: str, target_language: str) -> str:
        system = (
            "你是 Phoenix 本地医学翻译精修器。你的工作是核对英文原文并修正现有译文，"
            "不是总结。必须保持疾病、解剖、影像学、病理及检查技术术语准确；所有数字、"
            "单位、正负号、侧别、否定关系、分级、医学缩写、图表编号和诊断确定性必须与"
            "原文一致。不得删减、扩写或添加原文没有的医学知识。只输出完整修订译文。"
        )
        user = (
            f"目标语言：{target_language}\n\n"
            f"英文原文：\n{source}\n\n"
            f"现有译文：\n{draft}\n\n"
            "请输出修正后的完整医学译文。"
        )

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

    def refine(self, source: str, draft: str, target_language: str = "中文") -> str:
        source = str(source or "").strip()
        draft = str(draft or "").strip()
        if not source:
            raise ValueError("Local Qwen model3 requires a non-empty source")
        if not draft:
            raise ValueError("Local Qwen model3 is a refiner and requires a local draft")

        self._load()
        import torch

        if not self._first_refine_reported:
            print(
                "[Phoenix][模型3] 已进入实际翻译链：开始本地医学精修。",
                flush=True,
            )
            self._first_refine_reported = True

        prompt = self._prompt(source, draft, target_language)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1792,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])

        # Model3 is an editor, not a second full translator. Bound its output by
        # the existing Chinese draft so short paragraphs do not spend hundreds
        # of unnecessary generation tokens.
        max_new_tokens = max(160, min(512, int(len(draft) * 1.15) + 96))
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
                f"[Phoenix][模型3] 推理失败: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        generated = output[0][input_length:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("Local Qwen model3 returned empty text")

        elapsed = max(time.perf_counter() - started, 0.001)
        token_count = int(generated.numel())
        print(
            f"[Phoenix][模型3] 精修完成 | {elapsed:.1f}s | "
            f"{token_count / elapsed:.1f} token/s | {token_count} tokens",
            flush=True,
        )
        return text

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
