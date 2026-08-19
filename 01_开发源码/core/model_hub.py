from __future__ import annotations

import time

from core.hardware_profile import detect_hardware_profile


class ModelHub:

    HEAVY_3D_MODELS = {
        "monai_lung_nodule_ct",
    }

    def __init__(self):
        self.models = {}
        self.status = {}
        self.errors = {}
        self.load_ms = {}
        self.hardware_profile = detect_hardware_profile()

    def register(self, model):
        self.models[model.name] = model

    def _hardware_deferred(self, name):
        return (
            name in self.HEAVY_3D_MODELS
            and not self.hardware_profile.heavy_3d_allowed
        )

    def load_selected(self, names):
        for name in names:
            if self.status.get(name) == "loaded":
                continue

            if self._hardware_deferred(name):
                profile = self.hardware_profile
                self.status[name] = "hardware_deferred"
                self.errors[name] = (
                    "硬件保护：当前运行模式="
                    f"{profile.mode}，RAM={profile.ram_gb}GB，"
                    f"GPU={profile.gpu_name or '未识别'}。"
                    "monai_lung_nodule_ct 属于重型3D模型，"
                    "在医院8GB/K420工作站默认不在CPU上无限阻塞。"
                    "该状态不是阴性结果，疾病诊断有效性会保持为False。"
                    "如仅做受控性能测试，可显式设置 "
                    "PHOENIX_ALLOW_HEAVY_CPU=1 后重新启动。"
                )
                self.load_ms[name] = 0.0
                continue

            model = self.models.get(name)

            if model is None:
                self.status[name] = "missing"
                self.errors[name] = (
                    f"模型未注册: {name}"
                )
                continue

            started = time.perf_counter()

            try:
                model.load()
                self.status[name] = "loaded"
                self.errors.pop(name, None)

            except Exception as exc:
                self.status[name] = "failed"
                self.errors[name] = (
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                self.load_ms[name] = round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                )

    def predict_selected(self, case, names):
        results = {}

        for name in names:
            status = self.status.get(
                name,
                "not_loaded",
            )

            if status != "loaded":
                results[name] = {
                    "error": self.errors.get(
                        name,
                        f"模型未加载，status={status}",
                    ),
                    "stage": "load",
                    "status": status,
                    "load_ms": self.load_ms.get(name),
                }
                continue

            started = time.perf_counter()

            try:
                output = self.models[name].predict(case)
                elapsed_ms = round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                )

                if not isinstance(output, dict):
                    results[name] = {
                        "error": (
                            "模型predict返回值必须是dict，"
                            f"实际={type(output).__name__}"
                        ),
                        "stage": "predict",
                        "status": "failed",
                        "inference_ms": elapsed_ms,
                    }
                    continue

                output = dict(output)
                output.setdefault("model", name)
                output.setdefault("status", "success")
                output.setdefault("inference_ms", elapsed_ms)
                output.setdefault("load_ms", self.load_ms.get(name))
                results[name] = output

            except Exception as exc:
                elapsed_ms = round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                )
                results[name] = {
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "stage": "predict",
                    "status": "failed",
                    "inference_ms": elapsed_ms,
                    "load_ms": self.load_ms.get(name),
                }

        return results

    def unload_all(self):
        for model in self.models.values():
            try:
                model.unload()
            except Exception:
                pass

        self.status.clear()
        self.errors.clear()
        self.load_ms.clear()

    def summary(self):
        return {
            "hardware_profile": self.hardware_profile.to_dict(),
            "models": {
                name: {
                    "status": self.status.get(
                        name,
                        "not_loaded",
                    ),
                    "load_ms": self.load_ms.get(name),
                    "error": self.errors.get(name, ""),
                }
                for name in self.models
            },
        }

    def error_summary(self):
        return dict(self.errors)
