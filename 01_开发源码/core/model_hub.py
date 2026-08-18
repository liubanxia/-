from __future__ import annotations

import time


class ModelHub:

    def __init__(self):
        self.models = {}
        self.status = {}
        self.errors = {}
        self.load_ms = {}

    def register(self, model):
        self.models[model.name] = model

    def load_selected(self, names):
        for name in names:
            if self.status.get(name) == "loaded":
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
            name: {
                "status": self.status.get(
                    name,
                    "not_loaded",
                ),
                "load_ms": self.load_ms.get(name),
                "error": self.errors.get(name, ""),
            }
            for name in self.models
        }

    def error_summary(self):
        return dict(self.errors)
