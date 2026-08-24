from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable


CONTRACT_VERSION = "translation-output-v2"
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.%+\-/][A-Za-z0-9]+)*")


class TranslationOutputError(RuntimeError):
    pass


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def translation_digest(
    pages_root: Path,
    *,
    start_page: int,
    total_pages: int,
) -> str:
    digest = hashlib.sha256()
    root = Path(pages_root)
    for page_number in range(int(start_page), int(total_pages) + 1):
        page_file = root / f"{page_number:06d}.txt"
        if not page_file.is_file():
            raise TranslationOutputError(
                f"完整性检查失败：第 {page_number} 页译文checkpoint不存在。"
            )
        payload = page_file.read_bytes()
        digest.update(f"page:{page_number}\n".encode("ascii"))
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return "".join(
        ch.casefold()
        for ch in (text or "")
        if ch.isalnum() or "\u3400" <= ch <= "\u9fff"
    )


def _translation_probes(text: str) -> tuple[str, ...]:
    probes: list[str] = []
    for run in _CJK_RUN_RE.findall(text or ""):
        normalized = _normalized(run)
        if not normalized:
            continue
        if len(normalized) <= 10:
            probes.append(normalized)
        else:
            for start in range(0, len(normalized), 8):
                item = normalized[start : start + 8]
                if len(item) >= 3:
                    probes.append(item)
    for token in _LATIN_TOKEN_RE.findall(text or ""):
        normalized = _normalized(token)
        if len(normalized) >= 2:
            probes.append(normalized)
    return tuple(dict.fromkeys(probes))


def _probe_coverage(expected: str, actual: str) -> float:
    probes = _translation_probes(expected)
    if not probes:
        expected_norm = _normalized(expected)
        if not expected_norm:
            return 1.0
        return 1.0 if expected_norm in _normalized(actual) else 0.0
    actual_norm = _normalized(actual)
    hits = sum(1 for probe in probes if probe in actual_norm)
    return float(hits) / float(len(probes))


def validate_pdf(
    path: Path,
    *,
    expected_pages: int,
    pages_root: Path | None = None,
    start_page: int = 1,
    source_pdf: Path | None = None,
    preserve_source_images: bool = False,
    minimum_translation_coverage: float = 0.72,
) -> dict:
    import fitz

    path = Path(path)
    errors: list[str] = []
    if not path.is_file():
        raise TranslationOutputError(f"PDF成品不存在：{path}")
    file_size = int(path.stat().st_size)
    if file_size <= 0:
        raise TranslationOutputError(f"PDF成品为空文件：{path}")

    with path.open("rb") as handle:
        header = handle.read(8)
        seek = max(0, file_size - 8192)
        handle.seek(seek)
        tail = handle.read()
    if not header.startswith(b"%PDF-"):
        errors.append("缺少PDF文件头")
    if b"%%EOF" not in tail:
        errors.append("文件尾缺少%%EOF，疑似截断写入")

    expected_pages = max(1, int(expected_pages))
    source = None
    output = None
    page_manifest: list[dict] = []
    coverage_values: list[float] = []
    try:
        output = fitz.open(path)
        if bool(getattr(output, "needs_pass", False)):
            errors.append("输出PDF被意外加密")
        if int(output.page_count) != expected_pages:
            errors.append(
                f"页数不一致：期望 {expected_pages}，实际 {output.page_count}"
            )

        if preserve_source_images and source_pdf is not None:
            source = fitz.open(Path(source_pdf))

        page_limit = min(int(output.page_count), expected_pages)
        for output_index in range(page_limit):
            page = output[output_index]
            rect = page.rect
            if float(rect.width) <= 1.0 or float(rect.height) <= 1.0:
                errors.append(f"第 {output_index + 1} 页页面尺寸无效")

            try:
                image_count = len(page.get_images(full=True))
            except Exception:
                image_count = -1
                errors.append(f"第 {output_index + 1} 页图像资源无法读取")

            if source is not None:
                source_index = int(start_page) - 1 + output_index
                if 0 <= source_index < int(source.page_count):
                    try:
                        source_images = len(source[source_index].get_images(full=True))
                    except Exception:
                        source_images = -1
                    if source_images >= 0 and image_count >= 0 and image_count != source_images:
                        errors.append(
                            f"第 {output_index + 1} 页图像资源数变化："
                            f"源页 {source_images}，输出 {image_count}"
                        )

            try:
                extracted = page.get_text("text") or ""
            except Exception as exc:
                extracted = ""
                errors.append(
                    f"第 {output_index + 1} 页文字层读取失败：{type(exc).__name__}"
                )

            coverage = None
            if pages_root is not None:
                source_page_number = int(start_page) + output_index
                page_file = Path(pages_root) / f"{source_page_number:06d}.txt"
                if not page_file.is_file():
                    errors.append(
                        f"第 {source_page_number} 页译文checkpoint缺失"
                    )
                else:
                    expected_text = page_file.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).strip()
                    coverage = _probe_coverage(expected_text, extracted)
                    coverage_values.append(coverage)
                    if expected_text and coverage < float(minimum_translation_coverage):
                        errors.append(
                            f"第 {source_page_number} 页译文覆盖率仅 {coverage:.0%}，"
                            "疑似文字未写入或被截断"
                        )

            page_manifest.append(
                {
                    "page": output_index + 1,
                    "width": round(float(rect.width), 3),
                    "height": round(float(rect.height), 3),
                    "images": image_count,
                    "text_sha256": hashlib.sha256(
                        _normalized(extracted).encode("utf-8")
                    ).hexdigest(),
                    "translation_coverage": (
                        None if coverage is None else round(float(coverage), 4)
                    ),
                }
            )
    except Exception as exc:
        if isinstance(exc, TranslationOutputError):
            raise
        errors.append(f"PDF无法完整打开：{type(exc).__name__}: {exc}")
    finally:
        if output is not None:
            output.close()
        if source is not None:
            source.close()

    structural = hashlib.sha256(
        json.dumps(
            page_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "contract_version": CONTRACT_VERSION,
        "path": str(path),
        "bytes": file_size,
        "pages": len(page_manifest),
        "expected_pages": expected_pages,
        "translation_coverage_min": (
            round(min(coverage_values), 4) if coverage_values else None
        ),
        "translation_coverage_avg": (
            round(sum(coverage_values) / len(coverage_values), 4)
            if coverage_values
            else None
        ),
        "structure_sha256": structural,
        "file_sha256": sha256_file(path),
        "errors": errors,
        "passed": not errors,
        "page_manifest": page_manifest,
    }
    if errors:
        raise TranslationOutputError(
            "PDF成品完整性检查失败：" + "；".join(errors[:6])
        )
    return report


def validate_deliverables(
    paths: Iterable[Path],
    *,
    expected_complete_pages: int,
) -> dict:
    import fitz

    normalized = tuple(Path(path) for path in paths)
    if not normalized:
        raise TranslationOutputError("翻译任务没有生成任何交付文件。")
    seen: set[str] = set()
    items: list[dict] = []
    for index, path in enumerate(normalized):
        key = os.path.normcase(str(path.resolve()))
        if key in seen:
            raise TranslationOutputError(f"交付文件重复：{path}")
        seen.add(key)
        if not path.is_file():
            raise TranslationOutputError(f"交付文件不存在：{path}")
        size = int(path.stat().st_size)
        if size <= 0:
            raise TranslationOutputError(f"交付文件为空：{path}")
        suffix = path.suffix.lower()
        item = {
            "path": str(path),
            "suffix": suffix,
            "bytes": size,
            "sha256": sha256_file(path),
        }
        if suffix == ".pdf":
            try:
                doc = fitz.open(path)
                pages = int(doc.page_count)
                doc.close()
            except Exception as exc:
                raise TranslationOutputError(
                    f"PDF交付文件无法打开：{path.name}: {type(exc).__name__}: {exc}"
                ) from exc
            if pages <= 0:
                raise TranslationOutputError(f"PDF交付文件没有页面：{path.name}")
            if index == 0 and pages != int(expected_complete_pages):
                raise TranslationOutputError(
                    f"完整PDF页数异常：期望 {expected_complete_pages}，实际 {pages}"
                )
            item["pages"] = pages
        elif suffix == ".docx":
            try:
                with zipfile.ZipFile(path) as archive:
                    names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise TranslationOutputError(
                        f"DOCX结构不完整：{path.name}"
                    )
            except zipfile.BadZipFile as exc:
                raise TranslationOutputError(
                    f"DOCX交付文件损坏：{path.name}"
                ) from exc
        elif suffix in {".txt", ".md", ".html"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                raise TranslationOutputError(f"文本交付文件为空：{path.name}")
        items.append(item)

    digest = hashlib.sha256(
        json.dumps(
            items,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "contract_version": CONTRACT_VERSION,
        "passed": True,
        "count": len(items),
        "manifest_sha256": digest,
        "items": items,
    }


def build_input_signature(
    *,
    source_pdf: Path,
    pages_root: Path,
    start_page: int,
    total_pages: int,
    layout: str,
) -> dict:
    try:
        import fitz

        runtime = str(getattr(fitz, "VersionBind", "") or getattr(fitz, "__doc__", ""))
    except Exception:
        runtime = "unknown"
    return {
        "contract_version": CONTRACT_VERSION,
        "pdf_runtime": runtime[:160],
        "source_sha256": sha256_file(Path(source_pdf)),
        "translation_sha256": translation_digest(
            Path(pages_root),
            start_page=start_page,
            total_pages=total_pages,
        ),
        "start_page": int(start_page),
        "total_pages": int(total_pages),
        "layout": str(layout),
    }


def assert_stable_against_previous(
    report_path: Path,
    *,
    signature: dict,
    current_structure_sha256: str,
) -> None:
    report_path = Path(report_path)
    if not report_path.is_file():
        return
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return
    previous_signature = previous.get("input_signature") or {}
    if previous_signature != signature:
        return
    previous_structure = str(previous.get("structure_sha256", "") or "")
    if previous_structure and previous_structure != str(current_structure_sha256):
        raise TranslationOutputError(
            "相同源PDF、相同逐页译文和相同布局重复构建后结构摘要发生变化。"
            "Phoenix已拒绝覆盖上一份稳定成品。"
        )


def write_integrity_report(
    report_path: Path,
    *,
    signature: dict,
    pdf_report: dict,
    delivery_report: dict | None = None,
) -> None:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "passed": True,
        "input_signature": dict(signature),
        "structure_sha256": str(pdf_report.get("structure_sha256", "")),
        "pdf": dict(pdf_report),
    }
    if delivery_report is not None:
        payload["deliverables"] = dict(delivery_report)
    _atomic_write_json(Path(report_path), payload)
