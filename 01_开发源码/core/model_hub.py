class ModelHub:

    def __init__(self):
        self.models = {}

    def register(self, model):
        self.models[model.name] = model

    def load_all(self):
        for model in self.models.values():
            model.load()

    def predict_all(self, case):
        results = {}

        for name, model in self.models.items():
            results[name] = model.predict(case)

        return results

    def unload_all(self):
        for model in self.models.values():
            model.unload()

    def summary(self):
        return {
            name: self.status.get(
                name,
                "not_loaded",
            )
            for name in self.models
        }
