from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import chunk_text
from .config import WorkbenchPaths
from .llm import LocalLLM
from .pdf_parser import iter_pdf_pages, pdf_page_count, sha256_file
from .translation_models import MultiModelTranslationEngine


ProgressCallback = Callable[[int, int, str], None]


def _safe_filename(text: str) -> str:
    text = re.sub(r'[\\/:*?"<>|\r\n]+', '_', text).strip(' ._')
    return (text or '未命名PDF')[:120]


@dataclass(frozen=True)
class TranslationResult:
    output_path: Path
    source_path: Path
    start_page: int
    total_pages: int
    target_language: str
    pages_done: int
    resumed_pages: int
    warning_pages: int
    available_backends: tuple[str, ...]


class PDFTranslator:
    """Whole-book offline translator with multi-model fallback and checkpoints."""

    def __init__(self, paths: WorkbenchPaths, llm: LocalLLM):
        self.paths = paths
        self.llm = llm
        self.engine = MultiModelTranslationEngine(paths, llm)
        self.output_root = paths.evidence_root / 'PDF整本翻译'
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        temp.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _book_paths(
        self,
        pdf_path: Path,
        digest: str,
        target_language: str,
    ) -> tuple[Path, Path, Path, Path, Path]:
        book_name = _safe_filename(pdf_path.stem)
        language = _safe_filename(target_language)
        book_root = self.output_root / f'{book_name}_{digest[:10]}_{language}'
        pages_root = book_root / 'pages'
        audit_root = book_root / 'audit'
        checkpoint = book_root / 'checkpoint.json'
        final_output = book_root / f'{book_name}_完整译本_{language}.txt'
        pages_root.mkdir(parents=True, exist_ok=True)
        audit_root.mkdir(parents=True, exist_ok=True)
        return book_root, pages_root, audit_root, checkpoint, final_output

    def _translate_page(
        self,
        source_text: str,
        page_number: int,
        target_language: str,
    ) -> tuple[str, dict]:
        source_text = (source_text or '').strip()
        if not source_text:
            text = '[本页未提取到可翻译文字，可能是扫描页、图片页或需要OCR。]'
            return text, {
                'page': page_number,
                'warning_count': 1,
                'parts': [{
                    'part': 1,
                    'backend': 'none',
                    'quality_ok': False,
                    'quality_score': 0.0,
                    'reasons': ['PDF未提取到文字'],
                }],
            }

        # 1200 characters is deliberately conservative for the small dedicated
        # translation models. The general Qwen fallback can accept more, but a
        # single chunk size keeps all models comparable and avoids truncation.
        parts = chunk_text(
            source_text,
            max_chars=1200,
            overlap_chars=0,
        ) or [source_text]

        translated_parts: list[str] = []
        part_audits: list[dict] = []
        warning_count = 0

        for part_index, part_text in enumerate(parts, start=1):
            try:
                decision = self.engine.translate(part_text, target_language)
                translated = decision.text.strip()
                if decision.needs_review:
                    warning_count += 1
                    translated = (
                        f'[本段自动翻译需复核；使用模型={decision.backend}]\n'
                        + translated
                    )
                translated_parts.append(translated)
                part_audits.append({
                    'part': part_index,
                    'part_total': len(parts),
                    'backend': decision.backend,
                    'quality_ok': decision.quality.ok,
                    'quality_score': round(float(decision.quality.score), 4),
                    'reasons': list(decision.quality.reasons),
                    'needs_review': decision.needs_review,
                    'attempts': [
                        {
                            'backend': attempt.backend,
                            'quality_ok': attempt.quality.ok,
                            'quality_score': round(float(attempt.quality.score), 4),
                            'reasons': list(attempt.quality.reasons),
                        }
                        for attempt in decision.attempts
                    ],
                })
            except Exception as exc:
                # A single broken paragraph must never terminate a 1000-page
                # book. Preserve the exact source so it can be retried later.
                warning_count += 1
                translated_parts.append(
                    '[自动翻译失败；已保留原文，待补充模型后重试]\n' + part_text
                )
                part_audits.append({
                    'part': part_index,
                    'part_total': len(parts),
                    'backend': 'failed_all',
                    'quality_ok': False,
                    'quality_score': 0.0,
                    'reasons': [f'{type(exc).__name__}: {exc}'],
                    'needs_review': True,
                })

        return '\n\n'.join(translated_parts).strip(), {
            'page': page_number,
            'warning_count': warning_count,
            'parts': part_audits,
        }

    def _assemble_book(
        self,
        pdf_path: Path,
        pages_root: Path,
        output_path: Path,
        start_page: int,
        total_pages: int,
        target_language: str,
    ) -> None:
        sections = [
            f'{pdf_path.stem} - 完整{target_language}译本',
            f'原文件：{pdf_path}',
            f'翻译范围：第 {start_page} 页至第 {total_pages} 页',
            '',
        ]

        for page_number in range(start_page, total_pages + 1):
            page_file = pages_root / f'{page_number:06d}.txt'
            if not page_file.is_file():
                raise RuntimeError(
                    f'无法生成整本译本：第 {page_number} 页尚未完成翻译'
                )
            translated = page_file.read_text(encoding='utf-8').strip()
            sections.extend(
                [
                    f'===== 第 {page_number} 页 =====',
                    '',
                    translated,
                    '',
                ]
            )

        output_path.write_text('\n'.join(sections).rstrip() + '\n', encoding='utf-8')

    def translate_book(
        self,
        pdf_path: Path,
        *,
        start_page: int = 1,
        target_language: str = '中文',
        progress: ProgressCallback | None = None,
        force_restart: bool = False,
        retry_warning_pages: bool = False,
    ) -> TranslationResult:
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f'仅支持PDF整本翻译: {pdf_path}')

        backends = self.engine.available_backends()
        if not backends:
            raise RuntimeError(
                '整本PDF翻译没有可用本地模型。请至少安装 opus-mt-en-zh、'
                'NLLB-200-distilled-600M 或 Qwen3.5-4B 中的一个。'
            )

        total_pages = int(pdf_page_count(pdf_path))
        if total_pages <= 0:
            raise RuntimeError('PDF没有可读取页面')

        start_page = max(1, int(start_page))
        if start_page > total_pages:
            raise ValueError(
                f'开始页 {start_page} 超出PDF总页数 {total_pages}'
            )

        digest = sha256_file(pdf_path)
        book_root, pages_root, audit_root, checkpoint_path, final_output = self._book_paths(
            pdf_path,
            digest,
            target_language,
        )

        if force_restart and book_root.exists():
            for page_file in pages_root.glob('*.txt'):
                page_file.unlink(missing_ok=True)
            for audit_file in audit_root.glob('*.json'):
                audit_file.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)
            final_output.unlink(missing_ok=True)

        state = self._read_json(checkpoint_path)
        if state:
            if state.get('source_sha256') != digest:
                raise RuntimeError('检测到PDF内容已变化，请使用重新翻译模式。')
            if state.get('target_language') != target_language:
                raise RuntimeError('检测到目标语言已变化，请使用重新翻译模式。')
            if int(state.get('start_page', start_page)) != start_page:
                raise RuntimeError(
                    '该书已有不同开始页的翻译任务，请使用原开始页继续，'
                    '或选择重新翻译。'
                )

        state = {
            'source_path': str(pdf_path),
            'source_sha256': digest,
            'book_title': pdf_path.stem,
            'target_language': target_language,
            'start_page': start_page,
            'total_pages': total_pages,
            'status': 'running',
            'available_backends': backends,
            'last_completed_page': int(state.get('last_completed_page', start_page - 1)) if state else start_page - 1,
            'warning_pages': int(state.get('warning_pages', 0)) if state else 0,
        }
        self._write_json(checkpoint_path, state)

        selected_total = total_pages - start_page + 1
        pages_done = 0
        resumed_pages = 0
        warning_pages = 0

        try:
            for page_number, source_text in iter_pdf_pages(pdf_path):
                if page_number < start_page:
                    continue

                page_file = pages_root / f'{page_number:06d}.txt'
                audit_file = audit_root / f'{page_number:06d}.json'
                old_audit = self._read_json(audit_file)
                has_warning = int(old_audit.get('warning_count', 0)) > 0

                if (
                    page_file.is_file()
                    and page_file.stat().st_size > 0
                    and not (retry_warning_pages and has_warning)
                ):
                    pages_done += 1
                    resumed_pages += 1
                    if has_warning:
                        warning_pages += 1
                    state['last_completed_page'] = max(
                        int(state.get('last_completed_page', start_page - 1)),
                        page_number,
                    )
                    if progress:
                        progress(
                            pages_done,
                            selected_total,
                            f'第 {page_number}/{total_pages} 页已存在，跳过',
                        )
                    continue

                translated, audit = self._translate_page(
                    source_text,
                    page_number,
                    target_language,
                )
                page_file.write_text(translated.rstrip() + '\n', encoding='utf-8')
                self._write_json(audit_file, audit)

                pages_done += 1
                if int(audit.get('warning_count', 0)) > 0:
                    warning_pages += 1
                state['last_completed_page'] = page_number
                state['status'] = 'running'
                state['warning_pages'] = warning_pages
                state['available_backends'] = self.engine.available_backends()
                self._write_json(checkpoint_path, state)

                if progress:
                    backend_summary = ','.join(self.engine.available_backends())
                    progress(
                        pages_done,
                        selected_total,
                        f'整本翻译：已完成第 {page_number}/{total_pages} 页 '
                        f'| 模型={backend_summary} | 警告页={warning_pages}',
                    )

            self._assemble_book(
                pdf_path,
                pages_root,
                final_output,
                start_page,
                total_pages,
                target_language,
            )
            state['status'] = 'completed_with_warnings' if warning_pages else 'completed'
            state['last_completed_page'] = total_pages
            state['warning_pages'] = warning_pages
            state['output_path'] = str(final_output)
            self._write_json(checkpoint_path, state)

            return TranslationResult(
                output_path=final_output,
                source_path=pdf_path,
                start_page=start_page,
                total_pages=total_pages,
                target_language=target_language,
                pages_done=pages_done,
                resumed_pages=resumed_pages,
                warning_pages=warning_pages,
                available_backends=tuple(self.engine.available_backends()),
            )
        except Exception as exc:
            state['status'] = 'failed'
            state['error'] = f'{type(exc).__name__}: {exc}'
            self._write_json(checkpoint_path, state)
            raise
        finally:
            self.engine.unload()

    # Compatibility alias: all translation requests are whole-book jobs.
    def translate(self, pdf_path: Path, **kwargs) -> TranslationResult:
        kwargs.pop('end_page', None)
        return self.translate_book(pdf_path, **kwargs)
