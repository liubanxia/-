import gc
import torch

from ai_models.component_registry import PhoenixComponentRegistry


class ReportTeacherRuntime:
    def __init__(self, component_id):
        self.component_id = component_id
        self.registry = PhoenixComponentRegistry()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.status = "INTEGRATED_UNTESTED"

    def _path(self):
        item = self.registry.get(self.component_id)
        if not item:
            raise KeyError(self.component_id)
        return item["source_path"]

    def load(self, device="cpu"):
        import transformers

        path = self._path()

        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                path, local_files_only=True, trust_remote_code=True
            )
        except Exception:
            self.processor = None

        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = None

        classes = [
            getattr(transformers, "AutoModelForImageTextToText", None),
            getattr(transformers, "AutoModelForCausalLM", None),
            getattr(transformers, "AutoModelForSeq2SeqLM", None),
        ]

        last_error = None

        for cls in classes:
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(
                    path,
                    local_files_only=True,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    device_map=None,
                )
                break
            except Exception as e:
                last_error = e

        if self.model is None:
            raise RuntimeError(f"无法构造报告模型: {last_error}")

        self.model.eval()
        self.model.to(device)
        self.status = "LOADED_UNTESTED"
        return self

    def generate(self, prompt, max_new_tokens=512):
        if self.model is None:
            raise RuntimeError("report teacher not loaded")

        frontend = self.processor or self.tokenizer
        if frontend is None:
            raise RuntimeError("processor/tokenizer unavailable")

        try:
            inputs = frontend(
                text=prompt,
                return_tensors="pt",
            )
        except TypeError:
            inputs = frontend(
                prompt,
                return_tensors="pt",
            )

        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        decoder = self.tokenizer
        if decoder is None and self.processor is not None:
            decoder = getattr(self.processor, "tokenizer", None)

        if decoder is None:
            raise RuntimeError("decoder unavailable")

        return decoder.decode(ids[0], skip_special_tokens=True)

    def unload(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.status = "INTEGRATED_UNTESTED"
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
