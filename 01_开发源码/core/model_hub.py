class ModelHub:

    def __init__(self):
        self.models = {}
        self.status = {}

    def register(self, model):
        self.models[model.name] = model

    def load_selected(self, names):
        for name in names:
            if self.status.get(name) == "loaded":
                continue

            model = self.models.get(name)
            if model is None:
                continue

            try:
                model.load()
                self.status[name] = "loaded"
            except Exception as exc:
                self.status[name] = f"failed: {exc}"

    def predict_selected(self, case, names):
        results = {}

        for name in names:
            if self.status.get(name) != "loaded":
                continue

            try:
                results[name] = self.models[name].predict(case)
            except Exception as exc:
                results[name] = {"error": str(exc)}

        return results

    def unload_all(self):
        for model in self.models.values():
            try:
                model.unload()
            except Exception:
                pass

    def summary(self):
        return {
            name: self.status.get(name, "not_loaded")
            for name in self.models
        }
