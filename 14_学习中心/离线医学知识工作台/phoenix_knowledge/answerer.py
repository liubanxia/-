from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LocalLLM
from .retrieval import Evidence, Retriever


_CITATION_RE = re.compile(r"\[S(\d+)\]")


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
        parts = []
        for item in evidence:
            parts.append(
                f"{item.citation} 书名：{item.title}；页码：{item.page}\n{item.text}"
            )
        return "\n\n".join(parts)

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
            lines.append(
                f"- {item.citation} {item.title}，第{item.page}页"
            )
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
            "当前未加载本地生成模型。以下仅列出从PDF知识库检索到的原文证据，不做模型补充：",
            "",
        ]
        for item in evidence:
            excerpt = item.text.strip()
            if len(excerpt) > 700:
                excerpt = excerpt[:700] + "……"
            lines.append(
                f"{item.citation} {item.title} 第{item.page}页\n{excerpt}\n"
            )
        lines.append(self._source_footer(evidence).strip())
        return AnswerResult(
            text="\n".join(lines).strip(),
            evidence=evidence,
            mode="evidence_only",
        )

    def ask(
        self,
        query: str,
        *,
        limit: int = 18,
        use_embeddings: bool = True,
    ) -> AnswerResult:
        query = (query or "").strip()
        if not query:
            raise ValueError("问题不能为空")
        evidence = self.retriever.search(
            query,
            limit=limit,
            use_embeddings=use_embeddings,
        )
        if not evidence or not self.llm.available():
            return self._evidence_only(query, evidence)

        prompt = f"""你是 Phoenix 离线医学知识工作台。

绝对规则：
1. 只能依据下面给出的PDF证据回答，不得使用训练记忆、常识、互联网或未提供资料补充事实。
2. 每一个医学事实、数字、诊断标准、鉴别点、检查方法和建议都必须在句末标注对应证据编号，例如 [S12]。
3. 如果证据不足，明确写“当前导入资料中未找到明确依据”。
4. 不要编造页码、书名或来源编号。
5. 可以对多份证据做归纳、对照、去重，但不得改变原意。
6. 输出中文，优先按“核心结论—影像征象—鉴别诊断—陷阱/注意点—来源”组织；如果用户另有格式要求，以用户要求为准。

用户要求：
{query}

PDF证据：
{self._evidence_block(evidence)}
"""
        text = self.llm.generate(
            prompt,
            max_new_tokens=1600,
        ).strip()
        valid_ids = {item.chunk_id for item in evidence}
        used_ids = {
            int(x) for x in _CITATION_RE.findall(text)
        } & valid_ids

        if not used_ids:
            fallback = self._evidence_only(query, evidence)
            return AnswerResult(
                text=(
                    "本地生成模型返回的内容没有有效PDF引用，Phoenix已阻止该答案。\n\n"
                    + fallback.text
                ),
                evidence=evidence,
                mode="grounding_blocked",
            )

        return AnswerResult(
            text=text + self._source_footer(evidence, used_ids),
            evidence=evidence,
            mode="grounded_generation",
        )
