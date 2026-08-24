from __future__ import annotations


_INSTALLED = False
_FINAL_LOCAL_TAG = "|quality_final_v2"


def _remote_translation_selected(engine) -> bool:
    """Return True when the user selected a remote/API translation provider."""

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


def _is_local_final_backend(name: str) -> bool:
    value = str(name or "")
    if value.startswith("qwen_local_medical_model3"):
        return _FINAL_LOCAL_TAG in value
    return False


def _is_old_or_intermediate_local_backend(name: str) -> bool:
    value = str(name or "")
    return (
        value.startswith("marian")
        or value.startswith("nllb")
        or value.startswith("hymt15_1p8b")
        or (
            value.startswith("qwen_local_medical_model3")
            and _FINAL_LOCAL_TAG not in value
        )
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
            # Last stable release route: one Smart2 call per slide/paragraph
            # batch, then bounded retry only for rows that fail validation.
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

    # Route-aware checkpoint reuse. Switching from local123 to API must not
    # silently reuse local-final rows; switching back to local must not reuse
    # qwen35 API rows. This makes the first test after changing compute mode
    # genuinely exercise the selected route while preserving resume inside the
    # same route.
    try:
        from . import office_translation as office
        OfficeDocumentTranslator = office.OfficeDocumentTranslator
        original_translate_document = OfficeDocumentTranslator.translate_document

        def translate_document(self, *args, **kwargs):
            previous = getattr(self, "_phoenix_selected_translation_route", None)
            self._phoenix_selected_translation_route = (
                "api" if _remote_translation_selected(self.engine) else "local"
            )
            try:
                return original_translate_document(self, *args, **kwargs)
            finally:
                if previous is None:
                    try:
                        delattr(self, "_phoenix_selected_translation_route")
                    except Exception:
                        pass
                else:
                    self._phoenix_selected_translation_route = previous

        def load_completed_unit(
            self,
            path,
            unit,
            *,
            source_sha256: str,
            target_language: str,
            glossary_sha256: str,
        ):
            payload = office._read_json(path)
            if not payload or payload.get("source_sha256") != source_sha256:
                return None
            if payload.get("target_language") != target_language:
                return None
            if payload.get("glossary_sha256") != glossary_sha256:
                return None
            rows = payload.get("translations")
            if not isinstance(rows, list):
                return None

            expected = {segment.segment_id: segment.source for segment in unit.segments}
            translated: dict[str, str] = {}
            audits: list[dict] = []
            route = str(getattr(self, "_phoenix_selected_translation_route", "") or "")

            for row in rows:
                if not isinstance(row, dict):
                    return None
                segment_id = str(row.get("id", ""))
                if segment_id not in expected or row.get("source") != expected[segment_id]:
                    return None
                backend = str(row.get("backend", "") or "")

                if route == "api":
                    if _is_old_or_intermediate_local_backend(backend) or _is_local_final_backend(backend):
                        return None
                elif route == "local":
                    if backend.startswith("qwen35_medical_translation"):
                        return None
                    if _is_old_or_intermediate_local_backend(backend):
                        return None

                translated[segment_id] = str(row.get("translated", ""))
                audits.append(dict(row))

            if set(translated) != set(expected):
                return None
            warnings = int(payload.get("warning_count", 0) or 0)
            return translated, warnings, audits

        OfficeDocumentTranslator.translate_document = translate_document
        OfficeDocumentTranslator._load_completed_unit = load_completed_unit
    except Exception as exc:
        print(
            f"[Phoenix][翻译路线] 路线化checkpoint安装失败: {type(exc).__name__}: {exc}",
            flush=True,
        )

    print(
        "[Phoenix][翻译路线] 双路线已启用：选择API=上一稳定版Smart2批量精译；"
        "未选择/不可用=本地123。API不再是启动前提。",
        flush=True,
    )
