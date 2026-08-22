from __future__ import annotations

import os
from typing import Iterable


_INSTALLED = False


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.82))
    tail = max(1, limit - head - 24)
    return text[:head].rstrip() + "\n…[中间内容已按token预算截断]…\n" + text[-tail:].lstrip()


def _install_translation_smart1_offline() -> None:
    from .translation_models import MultiModelTranslationEngine, _normalize_smart_level

    original = MultiModelTranslationEngine.active_backends
    if getattr(original, "_phoenix_token_hardened", False):
        return

    def active_backends(self, target_language: str = "中文", smart_level: str = "smart1"):
        level = _normalize_smart_level(smart_level)
        backends = list(original(self, target_language, level))
        if level == "smart2":
            return backends

        # Smart1 is the low-cost/default translator. Preserve the existing
        # product/licensing/backend-selection rules, but remove the general LLM
        # backend so a normal whole-book translation cannot consume cloud tokens.
        return [backend for backend in backends if backend is not self.qwen]

    active_backends._phoenix_token_hardened = True
    MultiModelTranslationEngine.active_backends = active_backends


def _install_qa_prompt_budget() -> None:
    from .answerer import KnowledgeAnswerer

    original = KnowledgeAnswerer._evidence_block
    if getattr(original, "_phoenix_token_hardened", False):
        return

    def evidence_block(evidence):
        max_items = _env_int("PHOENIX_QA_MAX_EVIDENCE_ITEMS", 10, 3, 24)
        per_item = _env_int("PHOENIX_QA_MAX_CHARS_PER_EVIDENCE", 1200, 300, 3000)
        total_budget = _env_int("PHOENIX_QA_MAX_PROMPT_EVIDENCE_CHARS", 12000, 3000, 30000)
        parts: list[str] = []
        used = 0
        for item in list(evidence)[:max_items]:
            body = _clip(getattr(item, "text", ""), per_item)
            part = f"{item.citation} 资料：{item.title}；位置：第{item.page}页\n{body}"
            if parts and used + len(part) > total_budget:
                break
            if not parts and len(part) > total_budget:
                part = _clip(part, total_budget)
            parts.append(part)
            used += len(part)
        return "\n\n".join(parts)

    evidence_block._phoenix_token_hardened = True
    KnowledgeAnswerer._evidence_block = staticmethod(evidence_block)


def _install_organizer_prompt_budget() -> None:
    from .organizer import DeepOrganizer
    from .document_organizer import MultiDocumentOrganizer, _locator

    old_batch = DeepOrganizer._batch_prompt
    if getattr(old_batch, "_phoenix_token_hardened", False):
        return

    def batch_prompt(instruction: str, batch):
        max_items = _env_int("PHOENIX_ORGANIZE_MAX_EVIDENCE_PER_BATCH", 12, 4, 30)
        per_item = _env_int("PHOENIX_ORGANIZE_MAX_CHARS_PER_EVIDENCE", 800, 300, 2200)
        total_budget = _env_int("PHOENIX_ORGANIZE_BATCH_INPUT_CHARS", 10000, 4000, 24000)
        evidence_parts: list[str] = []
        used = 0
        for item in list(batch)[:max_items]:
            text = _clip(getattr(item, "text", ""), per_item)
            part = f"{item.citation} 书名：{item.title}；第{item.page}页\n{text}"
            if evidence_parts and used + len(part) > total_budget:
                break
            if not evidence_parts and len(part) > total_budget:
                part = _clip(part, total_budget)
            evidence_parts.append(part)
            used += len(part)
        evidence = "\n\n".join(evidence_parts)
        return f"""你正在处理 Phoenix 离线医学资料精确整理任务。
只能使用下面证据，不得调用任何外部知识。

整理规则：
1. 只保留与用户要求直接相关的内容，删除无关内容和重复表述。
2. 每条医学事实、数字、阈值、分级、检查技术、鉴别诊断必须保留 [S编号]。
3. 同一事实多来源合并并保留多个来源；冲突并列，不自行裁决。
4. 证据不足时明确说明，不得补写常识。
5. 不制造新的 S 编号，不改变原始数字和单位。
6. 输出紧凑，禁止复述任务说明、禁止空泛总结。

用户要求：{_clip(instruction, 1800)}

本批证据：
{evidence}
"""

    def multi_document_batch_prompt(instruction: str, batch):
        max_items = _env_int("PHOENIX_ORGANIZE_MAX_EVIDENCE_PER_BATCH", 12, 4, 30)
        per_item = _env_int("PHOENIX_ORGANIZE_MAX_CHARS_PER_EVIDENCE", 800, 300, 2200)
        total_budget = _env_int("PHOENIX_ORGANIZE_BATCH_INPUT_CHARS", 10000, 4000, 24000)
        parts: list[str] = []
        used = 0
        for item in list(batch)[:max_items]:
            body = _clip(getattr(item, "text", ""), per_item)
            part = f"{item.citation} 资料：{item.title}；{_locator(item)}\n{body}"
            if parts and used + len(part) > total_budget:
                break
            if not parts and len(part) > total_budget:
                part = _clip(part, total_budget)
            parts.append(part)
            used += len(part)
        evidence = "\n\n".join(parts)
        return f"""你正在处理 Phoenix 离线医学多资料精确整理任务。
只能使用下面证据，不得调用外部知识。

规则：
1. 只保留直接相关内容，删除重复和空话。
2. 每条医学事实、研究结果、数字、阈值和鉴别诊断必须保留 [S编号]。
3. 同义事实合并并保留多来源；冲突并列，不自行裁决。
4. 数字、单位、DOI/PMID、敏感度、特异度、AUC、置信区间和P值不得改变。
5. 保留页码、幻灯片号、文档/论文单元等来源定位；不得制造引用。
6. 输出紧凑，证据不足就明确说明。

用户要求：{_clip(instruction, 1800)}

本批证据：
{evidence}
"""

    def merge_prompt(title: str, instruction: str, partials: Iterable[str]):
        max_partials = _env_int("PHOENIX_ORGANIZE_MAX_PARTIALS_PER_MERGE", 6, 2, 8)
        per_partial = _env_int("PHOENIX_ORGANIZE_MAX_CHARS_PER_PARTIAL", 1800, 600, 4200)
        total_budget = _env_int("PHOENIX_ORGANIZE_MERGE_INPUT_CHARS", 10000, 4000, 24000)
        kept: list[str] = []
        used = 0
        for partial in list(partials)[:max_partials]:
            clipped = _clip(partial, per_partial)
            if kept and used + len(clipped) > total_budget:
                break
            if not kept and len(clipped) > total_budget:
                clipped = _clip(clipped, total_budget)
            kept.append(clipped)
            used += len(clipped)
        joined = "\n\n===== 分批笔记 =====\n".join(kept)
        return f"""你是 Phoenix 医学资料最终合并整理器。
只能基于下面已经带 [S编号] 的分批笔记重组，禁止加入新事实。

必须做到：
- 保留来源编号，不制造新的 S 编号。
- 删除重复、空话和偏离用户要求的内容。
- 同义内容合并；来源冲突并列。
- 每个关键结论紧跟来源。
- 输出紧凑，不复述输入，不为了篇幅扩写。

专题：{_clip(title, 300)}
用户要求：{_clip(instruction, 1600)}

{joined}
"""

    batch_prompt._phoenix_token_hardened = True
    multi_document_batch_prompt._phoenix_token_hardened = True
    merge_prompt._phoenix_token_hardened = True
    DeepOrganizer._batch_prompt = staticmethod(batch_prompt)
    MultiDocumentOrganizer._batch_prompt = staticmethod(multi_document_batch_prompt)
    DeepOrganizer._merge_prompt = staticmethod(merge_prompt)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_translation_smart1_offline()
    _install_qa_prompt_budget()
    _install_organizer_prompt_budget()
    _INSTALLED = True
