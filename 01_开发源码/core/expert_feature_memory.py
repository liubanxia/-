import threading


class ExpertFeatureMemory:

    def __init__(self):
        self._lock = threading.RLock()
        self._features = {}
        self._meta = {}

    def put(self, expert_id, tensor, **metadata):
        with self._lock:
            try:
                tensor = tensor.detach().cpu()
            except Exception:
                pass

            self._features[expert_id] = tensor
            self._meta[expert_id] = dict(metadata)

    def get(self, expert_id):
        with self._lock:
            return self._features.get(expert_id)

    def metadata(self):
        with self._lock:
            return dict(self._meta)

    def set_meta(self, key, value):
        with self._lock:
            self._meta[key] = value

    def clear(self):
        with self._lock:
            self._features.clear()
            self._meta.clear()


EXPERT_FEATURE_MEMORY = ExpertFeatureMemory()
