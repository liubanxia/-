from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import chunk_text
from .config import WorkbenchPaths
from .llm import LocalLLM


ProgressCallback = Callable[[int, int, str], None]


def _safe_filename(text: str) -> str:
    text = re.sub(r'[\\/:*?"<>|\r\n]+', '_', text).strip(' ._')
    return (text or '医学笔记')[:120]


@dataclass(frozen=True)
class NotesResult:
    output_path: Path
    text: str
    title: str
    chunks: int
    mode: str


class TXTNotesOrganizer:
    def __init__(self, paths: WorkbenchPaths, llm: LocalLLM):
        self.paths = paths
        self.llm = llm
        self.output_root = paths.evidence_root / 'TXT整理笔记'
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _batch_prompt(title: str, source_text: str, instruction: str, index: int, total: int) -> str:
        instruction = instruction.strip() or (
            '整理成适合医学影像学习和临床复习的结构化笔记：核心概念、影像征象、'
            '鉴别诊断、检查技术前提、诊断陷阱、报告表达和待补充问题。'
        )
        return f'''你是 Phoenix 离线医学TXT笔记整理器。

当前处理第 {index}/{total} 段。
只能整理下面用户提供的TXT/粘贴文字，不得调用外部知识，不得擅自补充事实。
可以去重、归类、重排、纠正明显排版，但不能改变医学原意。
数字、单位、分级、影像参数、疾病名、药名、缩写必须保留。
原文存在不确定、互相矛盾或缺失信息时必须保留并标记，不自行编造答案。
输出中文Markdown，适合长期保存、复习和打印。

笔记标题：{title}
整理要求：{instruction}

原始笔记：
{source_text}
'''

    @staticmethod
    def _merge_prompt(title: str, partials: list[str], instruction: str) -> str:
        joined = '\n\n===== 分段笔记 =====\n'.join(partials)
        return f'''你是 Phoenix 离线医学笔记总整理器。
只能根据下面已经整理好的分段笔记继续合并，不得加入新医学事实。
去除重复，统一术语，重排为清晰层级；所有关键数字、单位、条件、分级和缩写必须保留。
输出适合保存和打印的中文Markdown正文。

标题：{title}
整理要求：{instruction}

{joined}
'''

    def organize(
        self,
        source_text: str,
        *,
        title: str = '医学笔记',
        instruction: str = '',
        progress: ProgressCallback | None = None,
    ) -> NotesResult:
        source_text = (source_text or '').strip()
        title = (title or '医学笔记').strip()
        if not source_text:
            raise ValueError('TXT笔记内容不能为空')

        parts = chunk_text(source_text, max_chars=5000, overlap_chars=0) or [source_text]
        if not self.llm.available():
            final_text = f'# {title}\n\n{source_text}\n'
            mode = 'source_only'
        else:
            instruction = instruction.strip() or '整理成清晰、可复习、可打印的医学笔记'
            partials: list[str] = []
            for index, part in enumerate(parts, start=1):
                organized = self.llm.generate(
                    self._batch_prompt(title, part, instruction, index, len(parts)),
                    max_new_tokens=2200,
                ).strip()
                partials.append(organized or part)
                if progress:
                    progress(index, len(parts), f'已整理 {index}/{len(parts)} 段')

            level = partials
            while len(level) > 1:
                next_level: list[str] = []
                for start in range(0, len(level), 4):
                    group = level[start:start + 4]
                    if len(group) == 1:
                        next_level.append(group[0])
                        continue
                    merged = self.llm.generate(
                        self._merge_prompt(title, group, instruction),
                        max_new_tokens=3000,
                    ).strip()
                    next_level.append(merged or '\n\n'.join(group))
                level = next_level
            final_text = f'# {title}\n\n{level[0].strip()}\n'
            mode = 'local_ai'

        output_path = self.output_root / f'{_safe_filename(title)}.txt'
        output_path.write_text(final_text, encoding='utf-8')
        return NotesResult(
            output_path=output_path,
            text=final_text,
            title=title,
            chunks=len(parts),
            mode=mode,
        )

    def organize_file(
        self,
        path: Path,
        *,
        title: str | None = None,
        instruction: str = '',
        progress: ProgressCallback | None = None,
    ) -> NotesResult:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {'.txt', '.md'}:
            raise ValueError('只支持TXT/MD笔记文件')
        source = path.read_text(encoding='utf-8-sig', errors='replace')
        return self.organize(
            source,
            title=title or path.stem,
            instruction=instruction,
            progress=progress,
        )

    def save_text(self, text: str, title: str = '医学笔记') -> Path:
        text = (text or '').strip()
        if not text:
            raise ValueError('没有可保存的笔记内容')
        output_path = self.output_root / f'{_safe_filename(title)}.txt'
        output_path.write_text(text + '\n', encoding='utf-8')
        return output_path
