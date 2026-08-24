from __future__ import annotations


_INSTALLED = False


def _remote_translation_selected(engine) -> bool:
    """Return True when the user selected a remote/API translation provider.

    Merely having API credentials stored is not enough: Phoenix follows the
    compute mode selected in the workbench. If the remote route is selected but
    temporarily unavailable, the wrapper below falls back to the local 1->2->3
    chain instead of making API availability a hard requirement.
    """

    qwen = getattr(engine, "qwen", None)
    llm = getattr(qwen, "llm", None)
    if llm is None:
        return False

    try:
        compute = getattr(llm, "compute", None)
        if compute is not None and str(compute.requested_mode() or "").strip().lower() == "remote":
            return True
    except Exception:
        pass

    try:
        backend = str(llm.backend("translation") or "").strip().lower()
        return backend == "remote_server"
    except Exception:
        return False


def _report_route(engine, route: str) -> None:
    key = f"_phoenix_translation_route_reported_{route}"
    if bool(getattr(engine, key, False)):
        return
    setattr(engine, key, True)
    if route == "api_batch":
        label = ""
        try:
            llm = engine.qwen.llm
            label = str(llm.compute.provider_label() or "").strip()
        except Exception:
            label = ""
        suffix = f" | {label}" if label else ""
        print(
            "[Phoenix][翻译路线] 已选择API：恢复上一稳定版 Smart2 批量医学精译；"
            "每个单元一次批量调用，仅失败片段单独重试" + suffix,
            flush=True,
        )
    else:
        print(
            "[Phoenix][翻译路线] 未选择可用API：使用本地模型1→HY-MT模型2→Qwen模型3。",
            flush=True,
        )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .translation_models import MultiModelTranslationEngine, _normalize_smart_level

    cls = MultiModelTranslationEngine
    local_translate = cls.translate
    local_translate_segments = cls.translate_segments

    def translate(
        self,
        source: str,
        target_language: str = "中文",
        *,
        smart_level: str = "smart1",
    ):
        level = _normalize_smart_level(smart_level)
        if level == "smart2" and _remote_translation_selected(self):
            # hymt_cascade_policy stored the original validated Smart2 method
            # before the experimental/local cascade replaced it.
            base = getattr(self, "_phoenix_hymt_previous_translate", None)
            if callable(base):
                try:
                    _report_route(self, "api_batch")
                    return base(source, target_language, smart_level="smart2")
                except Exception as exc:
                    print(
                        f"[Phoenix][翻译路线] API单段精译不可用，自动回退本地123："
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        _report_route(self, "local")
        return local_translate(
            self,
            source,
            target_language,
            smart_level=level,
        )

    def translate_segments(
        self,
        sources: list[str],
        target_language: str = "中文",
        *,
        smart_level: str = "smart2",
    ):
        values = [str(value or "").strip() for value in sources]
        if not values:
            return ()
        level = _normalize_smart_level(smart_level)
        if level == "smart2" and _remote_translation_selected(self):
            # This is the route used by the last stable release: one Smart2
            # batch call for a slide/paragraph unit, followed only by bounded
            # per-row quality retries. It gives the remote model the untouched
            # English source instead of asking it to polish a degraded local
            # draft, while avoiding one API call per text box.
            base = getattr(self, "_phoenix_hymt_previous_translate_segments", None)
            if callable(base):
                try:
                    _report_route(self, "api_batch")
                    return base(values, target_language, smart_level="smart2")
                except Exception as exc:
                    print(
                        f"[Phoenix][翻译路线] API批量精译不可用，自动回退本地123："
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        _report_route(self, "local")
        return local_translate_segments(
            self,
            values,
            target_language,
            smart_level=level,
        )

    cls.translate = translate
    cls.translate_segments = translate_segments

    print(
        "[Phoenix][翻译路线] 双路线已启用：选择API=上一稳定版Smart2批量精译；"
        "未选择/不可用=本地123。API不再是启动前提。",
        flush=True,
    )
