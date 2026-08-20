from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .llm import LocalLLM
from .retrieval import Evidence, Retriever


_CITATION_RE = re.compile(r"\[S(\d+)\]")
_SCHOLARLY_FULLTEXT = {".html", ".htm", ".xml", ".nxml", ".jats"}
_SCHOLARLY_METADATA = {".nbib", ".ris", ".bib", ".bibtex", ".json", ".csljson"}
_CNKI_PAPER = {".caj", ".nh", ".hn", ".kdh", ".teb", ".c8"}


def _locator(item: Evidence) -> str:
    suffix = Path(item.path).suffix.lower()
    if suffix in {".ppt", ".pptx"}:
        return f"第{item.page}张幻灯片"
    if suffix == ".docx":
        return f"文档单元{item.page}"
    if suffix in {".txt", ".md"}:
        return f"文本单元{item.page}"
    if suffix in _SCHOLARLY_FULLTEXT:
        return f"论文单元{item.page}"
    if suffix in _SCHOLARLY_METADATA:
        return f"文献记录{item.page}"
    if suffix in _CNKI_PAPER:
        return f"第{item.page}页"
    return f"第{item.page}页"


@dataclass(frozen=True)
class AnswerResult:
    text: str
    evidence: list[Evidence]
    mode: str


class KnowledgeAnswerer:
    def __init__(self, retriever: Retriever, llm: LocalLLM):
        self.retriever = retriever
        self.llm = llm

    @staticmethod
    def _evidence_block(evidence: list[Evidence]) -> str:
        return "\n\n".join(
            f"{item.citation} 资料：{item.title}；位置：{_locator(item)}\n{item.text}"
            for item in evidence
        )

    @staticmethod
    def _source_footer(
        evidence: list[Evidence],
        used_ids: set[int] | None = None,
    ) -> str:
        selected = [
            item
            for item in evidence
            if used_ids is None or item.chunk_id in used_ids
        ]
        if not selected:
            return ""
        lines = ["\n\n---\n来源："]
        seen = set()
        for item in selected:
            key = (item.chunk_id, item.title, item.page)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {item.citation} {item.title}，{_locator(item)}")
        return "\n".join(lines)

    def _evidence_only(
        self,
        query: str,
        evidence: list[Evidence],
    ) -> AnswerResult:
        if not evidence:
            return AnswerResult(
                text="当前导入资料中未找到明确依据。",
                evidence=[],
                mode="evidence_only",
            )
        lines = [
            "快速资料问答：以下内容直接来自已导入资料；智能归纳属于可选第二阶段，不阻塞第一屏结果。",
            "",
        ]
        for item in evidence:
            excerpt = item.text.strip()
            if len(excerpt) > 900:
                excerpt = excerpt[:900] + "……"
            lines.append(
                f"{item.citation} {item.title} · {_locator(item)}\n{excerpt}\n"
            )
        lines.append(self._source_footer(evidence).strip())
        return AnswerResult(
            text="\n".join(lines).strip(),
            evidence=evidence,
            mode="evidence_only",
        )

    @staticmethod
    def _deep_enabled(explicit: bool | None) -> bool:
        if explicit is not None:
            return bool(explicit)
        raw = os.environ.get(
            "PHOENIX_KNOWLEDGE_DEEP_QA",
            "",
        ).strip().lower()
        if not raw:
            return True
        return raw in {"1", "true", "yes", "on"}

    def ask(
        self,
        query: str,
        *,
        limit: int = 18,
        use_embeddings: bool = True,
        deep: bool | None = None,
    ) -> AnswerResult:
        query = (query or "").strip()
        if not query:
            raise ValueError("问题不能为空")

        evidence = self.retriever.search(
            query,
            limit=limit,
            use_embeddings=use_embeddings,
        )
        if not evidence:
            return self._evidence_only(query, evidence)

        if not self._deep_enabled(deep) or not self.llm.available():
            return self._evidence_only(query, evidence)

        prompt = f"""你是 Phoenix 离线医学知识工作台。

绝对规则：
1. 只能依据下面给出的本地医学资料证据回答，不得使用训练记忆、常识、互联网或未提供资料补充事实。
2. 每一个医学事实、数字、诊断标准、鉴别点、检查方法和建议都必须在句末标注对应证据编号，例如 [S12]。
3. 如果证据不足，明确写“当前导入资料中未找到明确依据”。
4. 不要编造页码、幻灯片编号、论文单元、文献记录、书名或来源编号。
5. 可以对教材、课件、论文和题录证据做归纳、对照、去重，但不得改变原意。
6. 输出中文，优先按“核心结论—影像征象—鉴别诊断—研究证据—陷阱/注意点—来源”组织；如果用户另有格式要求，以用户要求为准。

用户要求：
{query}

本地资料证据：
{self._evidence_block(evidence)}
"""

        try:
            embeddings = getattr(
                self.retriever,
                "embeddings",
                None,
            )
            if embeddings is not None and hasattr(
                embeddings,
                "unload_model",
            ):
                embeddings.unload_model()
        except Exception:
            pass

        profile = os.environ.get(
            "PHOENIX_KNOWLEDGE_LLM_PROFILE",
            "fast",
        )
        max_tokens = (
            1600
            if profile.strip().lower()
            in {"deep", "4b", "deep4b", "quality", "max"}
            else 900
        )
        text = self.llm.generate(
            prompt,
            max_new_tokens=max_tokens,
        ).strip()
        valid_ids = {item.chunk_id for item in evidence}
        used_ids = {
            int(x)
            for x in _CITATION_RE.findall(text)
        } & valid_ids

        if not used_ids:
            fallback = self._evidence_only(query, evidence)
            return AnswerResult(
                text=(
                    "本地生成模型返回的内容没有有效资料引用，Phoenix已阻止该答案。\n\n"
                    + fallback.text
                ),
                evidence=evidence,
                mode="grounding_blocked",
            )

        return AnswerResult(
            text=text + self._source_footer(
                evidence,
                used_ids,
            ),
            evidence=evidence,
            mode="grounded_generation",
        )
