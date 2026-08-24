from __future__ import annotations

"""Speed up model3 without weakening its medical review role.

Model3 still reads the full English source, document context, terminology hints,
and the current local translation. The optimization is output-side only:
correct translations return a tiny PASS JSON; small errors return exact patches.
If the audit output is malformed, ambiguous, or fails Phoenix validation, the
existing full model3 refiner runs before API fallback is allowed.
"""

import json
from typing import Any

_INSTALLED = False
_MAX_PATCH_EDITS = 12


def _parse_audit_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型3终审未返回JSON对象")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型3终审JSON顶层必须是对象")
    return payload


def _apply_audit_payload(draft: str, payload: dict[str, Any]) -> tuple[str, str, int]:
    current = str(draft or "").strip()
    status = str(payload.get("status", "") or "").strip().upper()
    if status == "PASS":
        return current, "pass", 0

    if status == "FULL":
        final_text = str(payload.get("final_text", "") or "").strip()
        if not final_text:
            raise ValueError("FULL终审缺少final_text")
        return final_text, "full", 1

    if status not in {"PATCH", "FIX"}:
        raise ValueError(f"未知模型3终审状态: {status or '[empty]'}")

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("PATCH终审缺少edits")
    if len(edits) > _MAX_PATCH_EDITS:
        raise ValueError("PATCH修改项过多，改走模型3完整修订")

    updated = current
    applied = 0
    for row in edits:
        if not isinstance(row, dict):
            raise ValueError("PATCH修改项格式错误")
        old = str(row.get("old", "") or "")
        new = str(row.get("new", "") or "")
        if not old or old == new:
            raise ValueError("PATCH修改项为空或没有变化")
        occurrences = updated.count(old)
        if occurrences != 1:
            raise ValueError(
                f"PATCH原文定位不唯一({occurrences})，禁止猜测替换"
            )
        updated = updated.replace(old, new, 1)
        applied += 1

    if not updated.strip() or updated.strip() == current:
        raise ValueError("PATCH没有生成有效修订")
    return updated.strip(), "patch", applied


def _audit_prompt(self, source: str, draft: str, target_language: str) -> str:
    from . import translation_dual_route_release as contextual

    context = contextual._context()
    terms = "；".join(contextual._terms(source)) or "[无额外候选]"
    system = (
        "你是 Phoenix 本地模型3医学终审器。必须完整阅读英文原文、上一页/上一段上下文、"
        "医学术语提示和当前译文，然后逐句核对。重点检查疾病、解剖、影像征象、检查技术、"
        "病理、药物、统计术语、数字、单位、正负号、侧别、否定、分级、诊断确定性、缩写、"
        "图表编号以及漏译/错译。模型1和模型2可能翻错，因此不能只看中文流畅度。"
        "禁止总结、删减、扩写、解释、拒答或添加原文没有的信息。\n\n"
        "为了加速，只输出JSON：\n"
        "1) 译文完全正确：{\"status\":\"PASS\"}\n"
        "2) 少量错误且可精确定位：{\"status\":\"PATCH\",\"edits\":[{\"old\":\"译文中的原片段\",\"new\":\"修正片段\"}]}\n"
        "3) 错误较多、句法需要整体重写或无法用唯一old片段安全修改："
        "{\"status\":\"FULL\",\"final_text\":\"完整修订译文\"}\n"
        "PATCH中的old必须逐字复制当前译文中的唯一连续片段，禁止使用省略号。"
    )
    user = (
        f"目标语言：{target_language}\n"
        f"{context}"
        f"当前段医学英语术语/名词候选：{terms}\n\n"
        f"英文原文：\n{source}\n\n"
        f"当前本地译文：\n{draft}\n\n"
        "执行完整医学终审，只输出上述JSON对象。"
    )
    return self._chat_prompt(system, user)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .qwen_local_medical_backend import LocalQwenMedicalBackend
    from .translation_models import TranslationValidator

    if bool(getattr(LocalQwenMedicalBackend, "_phoenix_audit_acceleration", False)):
        return

    original_refine = LocalQwenMedicalBackend.refine
    LocalQwenMedicalBackend._audit_prompt = _audit_prompt

    def refine(self, source: str, draft: str, target_language: str = "中文") -> str:
        source = str(source or "").strip()
        draft = str(draft or "").strip()
        if not source:
            raise ValueError("Local Qwen model3 requires a non-empty source")
        if not draft:
            raise ValueError("Local Qwen model3 is a refiner and requires a local draft")

        if not bool(getattr(self, "_phoenix_audit_mode_reported", False)):
            print(
                "[Phoenix][模型3] 加速终审：完整读取原文/上下文/术语；"
                "正确项仅PASS，少量错误输出PATCH；不减少审核深度。",
                flush=True,
            )
            self._phoenix_audit_mode_reported = True

        try:
            self._load()
            prompt = self._audit_prompt(source, draft, target_language)
            raw = self._generate_prompt(
                prompt,
                draft,
                mode_label="医学终审审计",
                max_input_length=3072,
                max_output_tokens=320,
            )
            payload = _parse_audit_payload(raw)
            candidate, mode, edit_count = _apply_audit_payload(draft, payload)

            # Never let a short-output optimization weaken the existing safety
            # gate. PASS/PATCH must independently satisfy Phoenix validation.
            report = TranslationValidator().validate(
                source,
                candidate,
                target_language,
            )
            if report.ok:
                if mode == "pass":
                    print(
                        "[Phoenix][模型3] 终审=PASS，保留前级译文，跳过全文重复生成。",
                        flush=True,
                    )
                elif mode == "patch":
                    print(
                        f"[Phoenix][模型3] 终审=PATCH，已本地应用 {edit_count} 处修正。",
                        flush=True,
                    )
                else:
                    print(
                        "[Phoenix][模型3] 终审=FULL，已按需生成完整修订。",
                        flush=True,
                    )
                return candidate

            print(
                "[Phoenix][模型3] 审计候选未通过质量门，先回退模型3完整修订；不直接消耗API。",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Phoenix][模型3] 审计式终审无法安全应用: {type(exc).__name__}: {exc}；"
                "回退模型3完整修订。",
                flush=True,
            )

        # Accuracy-first fallback: preserve the proven full model3 correction.
        # API remains downstream and is reached only if this full local repair
        # also fails the caller's quality gate.
        return original_refine(self, source, draft, target_language)

    LocalQwenMedicalBackend.refine = refine
    LocalQwenMedicalBackend._phoenix_audit_acceleration = True

    print(
        "[Phoenix][模型3加速] 已启用PASS/PATCH终审：模型3仍完整审核上下文；"
        "只有需要时才生成长译文，异常自动回退原完整修订后才允许API兜底。",
        flush=True,
    )
