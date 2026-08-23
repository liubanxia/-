from __future__ import annotations

import time
from core.hardware_profile import detect_hardware_profile


class ModelHub:
    HEAVY_3D_MODELS = {"monai_lung_nodule_ct"}

    def __init__(self):
        self.models, self.status, self.errors, self.load_ms = {}, {}, {}, {}
        self.hardware_profile = detect_hardware_profile()

    def register(self, model): self.models[model.name] = model

    def load_selected(self, names):
        for name in names:
            if self.status.get(name) == "loaded": continue
            if name in self.HEAVY_3D_MODELS and not self.hardware_profile.heavy_3d_allowed:
                self.status[name] = "hardware_deferred"
                self.errors[name] = "Heavy 3D model deferred by hardware protection policy; this is not a negative result."
                self.load_ms[name] = 0.0
                continue
            model = self.models.get(name)
            if model is None:
                self.status[name] = "missing"; self.errors[name] = f"模型未注册: {name}"; continue
            started = time.perf_counter()
            try:
                model.load(); self.status[name] = "loaded"; self.errors.pop(name, None)
            except Exception as exc:
                self.status[name] = "failed"; self.errors[name] = f"{type(exc).__name__}: {exc}"
            self.load_ms[name] = round((time.perf_counter() - started) * 1000, 3)

    def predict_selected(self, case, names):
        results = {}
        for name in names:
            status = self.status.get(name, "not_loaded")
            if status != "loaded":
                results[name] = {"error": self.errors.get(name, f"模型未加载，status={status}"), "stage": "load", "status": status, "load_ms": self.load_ms.get(name)}
                continue
            started = time.perf_counter()
            try:
                output = self.models[name].predict(case)
                elapsed = round((time.perf_counter() - started) * 1000, 3)
                if not isinstance(output, dict):
                    results[name] = {"error": f"模型predict返回值必须是dict，实际={type(output).__name__}", "stage": "predict", "status": "failed", "inference_ms": elapsed}
                else:
                    output = dict(output); output.setdefault("model", name); output.setdefault("status", "success"); output.setdefault("inference_ms", elapsed); output.setdefault("load_ms", self.load_ms.get(name)); results[name] = output
            except Exception as exc:
                results[name] = {"error": f"{type(exc).__name__}: {exc}", "stage": "predict", "status": "failed", "inference_ms": round((time.perf_counter() - started) * 1000, 3), "load_ms": self.load_ms.get(name)}
        return results

    def unload_all(self):
        for model in self.models.values():
            try: model.unload()
            except Exception: pass
        self.status.clear(); self.errors.clear(); self.load_ms.clear()
