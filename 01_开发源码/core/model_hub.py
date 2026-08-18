class ModelHub:

    def __init__(self):
        self.models = {}
        self.status = {}
        self.errors = {}

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

            try:
                model.load()
                self.status[name] = "loaded"
                self.errors.pop(name, None)

            except Exception as exc:
                self.status[name] = "failed"
                self.errors[name] = (
                    f"{type(exc).__name__}: {exc}"
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
                }
                continue

            try:
                results[name] = (
                    self.models[name].predict(case)
                )

            except Exception as exc:
                results[name] = {
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "stage": "predict",
                    "status": "failed",
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

    def summary(self):
        return {
            name: self.status.get(
                name,
                "not_loaded",
            )
            for name in self.models
        }

    def error_summary(self):
        return dict(self.errors)
