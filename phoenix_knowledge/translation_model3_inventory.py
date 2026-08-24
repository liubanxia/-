from __future__ import annotations

from .translation_cascade_v2 import _model3, _model3_available
from .translation_models import MultiModelTranslationEngine, _normalize_smart_level


def _with_model3(engine: MultiModelTranslationEngine, backends: list[object]) -> list[object]:
    if not _model3_available(engine):
        return backends

    backend = _model3(engine)
    name = str(getattr(backend, "name", "") or "").strip()
    if not name:
        return backends
    if any(str(getattr(item, "name", "") or "").strip() == name for item in backends):
        return backends

    # Keep the visible experimental order: HY-MT -> local Qwen model3 ->
    # Smart2 API. Legacy Smart1 backends are never formal candidates.
    insert_at = len(backends)
    for index, item in enumerate(backends):
        item_name = str(getattr(item, "name", "") or "").strip()
        if item_name.startswith("qwen35_medical_translation"):
            insert_at = index
            break
    backends.insert(insert_at, backend)
    return backends


def install() -> None:
    cls = MultiModelTranslationEngine
    if bool(getattr(cls, "_phoenix_model3_inventory_installed", False)):
        return

    previous_active = cls.active_backends
    previous_available = cls.available_backends

    def active_backends(
        self: MultiModelTranslationEngine,
        target_language: str = "中文",
        smart_level: str = "smart1",
    ) -> list[object]:
        result = list(previous_active(self, target_language, smart_level))
        if _normalize_smart_level(smart_level) == "smart2":
            result = _with_model3(self, result)
        return result

    def available_backends(self: MultiModelTranslationEngine) -> list[str]:
        names = list(previous_available(self))
        if _model3_available(self):
            model3_name = str(getattr(_model3(self), "name", "") or "").strip()
            if model3_name:
                # Put model3 before Smart2 in the visible inventory.
                try:
                    index = next(
                        i for i, name in enumerate(names)
                        if str(name).startswith("qwen35_medical_translation")
                    )
                except StopIteration:
                    index = len(names)
                if model3_name not in names:
                    names.insert(index, model3_name)
        return list(dict.fromkeys(str(name) for name in names if str(name)))

    def formal_backend_names(
        self: MultiModelTranslationEngine,
        target_language: str = "中文",
    ) -> list[str]:
        names = [
            str(getattr(backend, "name", "") or "").strip()
            for backend in self.active_backends(target_language, "smart2")
        ]
        return list(dict.fromkeys(name for name in names if name))

    cls.active_backends = active_backends
    cls.available_backends = available_backends
    cls.formal_backend_names = formal_backend_names
    cls._phoenix_model3_inventory_installed = True
