from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import chunk_text
from .config import WorkbenchPaths
from .llm import LocalLLM
from .pdf_assets import PDFAssetStore, html_images, markdown_images
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
    output_paths: tuple[Path, ...] = ()
    image_count: int = 0


class PDFTranslator:
    """Whole-book offline translator with checkpoints, rich media and multi-format output."""

    def __init__(self, paths: WorkbenchPaths, llm: LocalLLM):
        self.paths = paths
        self.llm = llm
        self.engine = MultiModelTranslationEngine(paths, llm)
        self.assets = PDFAssetStore(paths.runtime_root)
        self.output_root = paths.evidence_root / 'PDF整本翻译'
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    @staticmethod
    def _resolve_resume_start_page(
        state: dict,
        requested_start_page: int,
        *,
        force_restart: bool = False,
    ) -> int:
        requested = max(1, int(requested_start_page))
        if force_restart or not state:
            return requested
        try:
            existing = int(state.get('start_page', requested))
        except (TypeError, ValueError):
            return requested
        return max(1, existing)

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
        *,
        status: Callable[[str], None] | None = None,
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

        parts = chunk_text(source_text, max_chars=1200, overlap_chars=0) or [source_text]
        translated_parts: list[str] = []
        part_audits: list[dict] = []
        warning_count = 0
        active_names = [x.name for x in self.engine.active_backends(target_language)]

        for part_index, part_text in enumerate(parts, start=1):
            if status:
                status(
                    f'第 {page_number} 页：正在翻译第 {part_index}/{len(parts)} 段 '
                    f'| 模型={" → ".join(active_names) if active_names else "无"}'
                )
            try:
                decision = self.engine.translate(part_text, target_language)
                translated = decision.text.strip()
                if decision.needs_review:
                    warning_count += 1
                    translated = (
                        f'[本段自动翻译需复核；使用模型={decision.backend}]\n' + translated
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
                if status:
                    status(
                        f'第 {page_number} 页：第 {part_index}/{len(parts)} 段完成 '
                        f'| 使用={decision.backend} | 质量={decision.quality.score:.2f}'
                    )
            except Exception as exc:
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
                if status:
                    status(f'第 {page_number} 页：第 {part_index}/{len(parts)} 段失败，已保留原文继续')

        return '\n\n'.join(translated_parts).strip(), {
            'page': page_number,
            'warning_count': warning_count,
            'parts': part_audits,
        }

    def _assemble_outputs(
        self,
        pdf_path: Path,
        book_root: Path,
        pages_root: Path,
        start_page: int,
        total_pages: int,
        target_language: str,
    ) -> tuple[tuple[Path, ...], int]:
        book_name = _safe_filename(pdf_path.stem)
        language = _safe_filename(target_language)
        txt_path = book_root / f'{book_name}_完整译本_{language}.txt'
        md_path = book_root / f'{book_name}_完整译本_{language}.md'
        html_path = book_root / f'{book_name}_完整译本_{language}.html'
        docx_path = book_root / f'{book_name}_完整译本_{language}.docx'
        images_root = book_root / 'images'
        images_root.mkdir(parents=True, exist_ok=True)

        txt_sections = [
            f'{pdf_path.stem} - 完整{target_language}译本',
            f'原文件：{pdf_path}',
            f'翻译范围：第 {start_page} 页至第 {total_pages} 页',
            '',
        ]
        md_sections = [
            f'# {pdf_path.stem} - 完整{target_language}译本',
            '',
            f'- 原文件：`{pdf_path}`',
            f'- 翻译范围：第 {start_page} 页至第 {total_pages} 页',
            '',
        ]
        html_sections = [
            '<!doctype html><html><head><meta charset="utf-8">',
            f'<title>{html.escape(pdf_path.stem)} - 完整{html.escape(target_language)}译本</title>',
            '<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1000px;margin:2em auto;line-height:1.65;padding:0 1em}pre{white-space:pre-wrap}figure{margin:1.2em 0}img{border:1px solid #ddd}</style>',
            '</head><body>',
            f'<h1>{html.escape(pdf_path.stem)} - 完整{html.escape(target_language)}译本</h1>',
            f'<p>原文件：{html.escape(str(pdf_path))}</p>',
            f'<p>翻译范围：第 {start_page} 页至第 {total_pages} 页</p>',
        ]

        docx = None
        try:
            from docx import Document
            docx = Document()
            docx.add_heading(f'{pdf_path.stem} - 完整{target_language}译本', 0)
            docx.add_paragraph(f'原文件：{pdf_path}')
            docx.add_paragraph(f'翻译范围：第 {start_page} 页至第 {total_pages} 页')
        except Exception:
            docx = None

        image_count = 0
        for page_number in range(start_page, total_pages + 1):
            page_file = pages_root / f'{page_number:06d}.txt'
            if not page_file.is_file():
                raise RuntimeError(f'无法生成整本译本：第 {page_number} 页尚未完成翻译')
            translated = page_file.read_text(encoding='utf-8').strip()
            copied = self.assets.copy_page_assets(
                pdf_path,
                page_number,
                images_root,
                prefix='',
                ensure=True,
            )
            image_count += len(copied)

            txt_sections.extend([f'===== 第 {page_number} 页 =====', '', translated, ''])
            md_sections.extend([f'## 第 {page_number} 页', '', translated, ''])
            if copied:
                md_sections.append(
                    markdown_images(copied, relative_to=book_root, label=f'第{page_number}页原图')
                )

            html_sections.extend([
                f'<h2>第 {page_number} 页</h2>',
                f'<pre>{html.escape(translated)}</pre>',
            ])
            if copied:
                html_sections.append(
                    html_images(copied, relative_to=book_root, label=f'第{page_number}页原图')
                )

            if docx is not None:
                try:
                    docx.add_heading(f'第 {page_number} 页', level=1)
                    docx.add_paragraph(translated)
                    for image_path in copied:
                        try:
                            docx.add_picture(str(image_path))
                        except Exception:
                            docx.add_paragraph(f'[图片文件：{image_path.name}]')
                except Exception:
                    pass

        html_sections.append('</body></html>')
        txt_path.write_text('\n'.join(txt_sections).rstrip() + '\n', encoding='utf-8')
        md_path.write_text('\n'.join(md_sections).rstrip() + '\n', encoding='utf-8')
        html_path.write_text('\n'.join(html_sections), encoding='utf-8')
        outputs: list[Path] = [txt_path, md_path, html_path]
        if docx is not None:
            try:
                docx.save(str(docx_path))
                outputs.append(docx_path)
            except Exception:
                pass
        return tuple(outputs), image_count

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

        active_backends = self.engine.active_backends(target_language)
        if not active_backends:
            raise RuntimeError(f'目标语言“{target_language}”当前没有可用本地翻译模型。')

        total_pages = int(pdf_page_count(pdf_path))
        if total_pages <= 0:
            raise RuntimeError('PDF没有可读取页面')

        start_page = max(1, int(start_page))
        if start_page > total_pages:
            raise ValueError(f'开始页 {start_page} 超出PDF总页数 {total_pages}')

        digest = sha256_file(pdf_path)
        book_root, pages_root, audit_root, checkpoint_path, final_output = self._book_paths(
            pdf_path, digest, target_language
        )

        if force_restart and book_root.exists():
            for page_file in pages_root.glob('*.txt'):
                page_file.unlink(missing_ok=True)
            for audit_file in audit_root.glob('*.json'):
                audit_file.unlink(missing_ok=True)
            for pattern in ('*.txt', '*.md', '*.html', '*.docx'):
                for output in book_root.glob(pattern):
                    output.unlink(missing_ok=True)
            shutil.rmtree(book_root / 'images', ignore_errors=True)
            checkpoint_path.unlink(missing_ok=True)

        state = self._read_json(checkpoint_path)
        if state:
            if state.get('source_sha256') != digest:
                raise RuntimeError('检测到PDF内容已变化，请使用重新翻译模式。')
            if state.get('target_language') != target_language:
                raise RuntimeError('检测到目标语言已变化，请使用重新翻译模式。')

        start_page = self._resolve_resume_start_page(state, start_page, force_restart=force_restart)
        if start_page > total_pages:
            raise RuntimeError(f'翻译checkpoint中的开始页 {start_page} 超出PDF总页数 {total_pages}')

        if progress:
            progress(0, total_pages - start_page + 1, '正在读取PDF图片资料并准备翻译……')
        try:
            self.assets.extract(pdf_path)
        except Exception:
            pass

        all_backends = self.engine.available_backends()
        active_names = [x.name for x in self.engine.active_backends(target_language)]
        state = {
            'source_path': str(pdf_path),
            'source_sha256': digest,
            'book_title': pdf_path.stem,
            'target_language': target_language,
            'start_page': start_page,
            'total_pages': total_pages,
            'status': 'running',
            'available_backends': all_backends,
            'active_backends': active_names,
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

                if page_file.is_file() and page_file.stat().st_size > 0 and not (retry_warning_pages and has_warning):
                    pages_done += 1
                    resumed_pages += 1
                    if has_warning:
                        warning_pages += 1
                    state['last_completed_page'] = max(int(state.get('last_completed_page', start_page - 1)), page_number)
                    if progress:
                        progress(pages_done, selected_total, f'第 {page_number}/{total_pages} 页已存在，跳过')
                    continue

                def note(message: str) -> None:
                    if progress:
                        progress(pages_done, selected_total, message)

                translated, audit = self._translate_page(
                    source_text,
                    page_number,
                    target_language,
                    status=note,
                )
                page_file.write_text(translated.rstrip() + '\n', encoding='utf-8')
                self._write_json(audit_file, audit)

                pages_done += 1
                if int(audit.get('warning_count', 0)) > 0:
                    warning_pages += 1
                state['last_completed_page'] = page_number
                state['status'] = 'running'
                state['warning_pages'] = warning_pages
                self._write_json(checkpoint_path, state)
                if progress:
                    progress(
                        pages_done,
                        selected_total,
                        f'整本翻译：已完成第 {page_number}/{total_pages} 页 | 警告页={warning_pages}',
                    )

            if progress:
                progress(selected_total, selected_total, '翻译页已完成，正在生成 TXT / Markdown / HTML / DOCX 与图片目录……')
            output_paths, image_count = self._assemble_outputs(
                pdf_path,
                book_root,
                pages_root,
                start_page,
                total_pages,
                target_language,
            )
            final_output = output_paths[0]
            state['status'] = 'completed_with_warnings' if warning_pages else 'completed'
            state['last_completed_page'] = total_pages
            state['warning_pages'] = warning_pages
            state['output_path'] = str(final_output)
            state['output_paths'] = [str(x) for x in output_paths]
            state['image_count'] = image_count
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
                available_backends=tuple(all_backends),
                output_paths=output_paths,
                image_count=image_count,
            )
        except Exception as exc:
            state['status'] = 'failed'
            state['error'] = f'{type(exc).__name__}: {exc}'
            self._write_json(checkpoint_path, state)
            raise
        finally:
            self.engine.unload()

    def translate(self, pdf_path: Path, **kwargs) -> TranslationResult:
        kwargs.pop('end_page', None)
        return self.translate_book(pdf_path, **kwargs)
