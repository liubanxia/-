from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path

CONTRACT_VERSION = 2


class OutputContractError(RuntimeError):
    pass


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _required_file(path: Path, label: str, minimum: int = 1) -> Path:
    path = Path(path)
    if not path.is_file():
        raise OutputContractError(f"{label}不存在：{path}")
    size = int(path.stat().st_size)
    if size < minimum:
        raise OutputContractError(f"{label}为空或过小：{path} ({size} bytes)")
    return path


def validate_text_file(path: Path, *, label: str = "文本成品") -> dict:
    path = _required_file(path, label)
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise OutputContractError(f"{label}含NUL字节，疑似损坏：{path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OutputContractError(f"{label}不是有效UTF-8：{path}: {exc}") from exc
    if not text.strip():
        raise OutputContractError(f"{label}没有有效内容：{path}")
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "chars": len(text),
    }


def validate_docx(path: Path) -> dict:
    path = _required_file(path, "DOCX成品", minimum=64)
    try:
        with zipfile.ZipFile(path) as archive:
            broken = archive.testzip()
            if broken:
                raise OutputContractError(f"DOCX压缩包损坏：{broken}")
            names = set(archive.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            missing = sorted(required - names)
            if missing:
                raise OutputContractError("DOCX缺少关键结构：" + ", ".join(missing))
            document_xml = archive.read("word/document.xml")
            if len(document_xml) < 32:
                raise OutputContractError("DOCX正文结构异常短")
    except OutputContractError:
        raise
    except Exception as exc:
        raise OutputContractError(f"DOCX无法重新打开：{path}: {exc}") from exc
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "structure_sha256": hashlib.sha256(document_xml).hexdigest(),
    }


def validate_pdf(path: Path) -> dict:
    import fitz

    path = _required_file(path, "PDF成品", minimum=64)
    raw_head = path.read_bytes()[:8]
    if not raw_head.startswith(b"%PDF-"):
        raise OutputContractError(f"PDF文件头无效：{path}")
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - 4096))
        tail = handle.read()
    if b"%%EOF" not in tail:
        raise OutputContractError(f"PDF缺少EOF标记，疑似截断：{path}")

    structure = []
    text_chars = 0
    image_count = 0
    try:
        doc = fitz.open(path)
        try:
            if getattr(doc, "needs_pass", False):
                raise OutputContractError(f"PDF成品被加密且无法直接验收：{path}")
            if int(doc.page_count) <= 0:
                raise OutputContractError(f"PDF没有页面：{path}")
            for index in range(doc.page_count):
                page = doc[index]
                text = page.get_text("text") or ""
                images = page.get_images(full=True)
                text_chars += len(text)
                image_count += len(images)
                structure.append({
                    "page": index + 1,
                    "text_sha256": hashlib.sha256(
                        text.encode("utf-8", errors="replace")
                    ).hexdigest(),
                    "text_chars": len(text),
                    "images": len(images),
                })
        finally:
            doc.close()
    except OutputContractError:
        raise
    except Exception as exc:
        raise OutputContractError(f"PDF无法重新打开：{path}: {exc}") from exc

    if text_chars <= 0 and image_count <= 0:
        raise OutputContractError(f"PDF既无文字也无图像：{path}")
    structure_sha = hashlib.sha256(
        json.dumps(structure, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "pages": len(structure),
        "text_chars": text_chars,
        "images": image_count,
        "structure_sha256": structure_sha,
    }


_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((.+)\)")


def _validate_markdown_assets(path: Path) -> int:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    checked = 0
    for raw in text.splitlines():
        match = _IMAGE_RE.fullmatch(raw.strip())
        if not match:
            continue
        reference = match.group(1).strip().strip("<>")
        if reference.startswith(("http://", "https://", "data:")):
            continue
        target = Path(reference)
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file() or target.stat().st_size <= 0:
            raise OutputContractError(
                f"Markdown引用的本地图像不存在/为空：{reference}"
            )
        checked += 1
    return checked


def validate_export_bundle(bundle) -> dict:
    markdown = validate_text_file(Path(bundle.markdown), label="Markdown成品")
    markdown["local_images"] = _validate_markdown_assets(Path(bundle.markdown))
    text = validate_text_file(Path(bundle.text), label="TXT成品")
    docx = validate_docx(Path(bundle.docx))
    pdf = validate_pdf(Path(bundle.pdf))
    structure_payload = {
        "markdown": markdown["sha256"],
        "text": text["sha256"],
        "docx_structure": docx["structure_sha256"],
        "pdf_structure": pdf["structure_sha256"],
    }
    return {
        "contract": CONTRACT_VERSION,
        "markdown": markdown,
        "text": text,
        "docx": docx,
        "pdf": pdf,
        "structure_sha256": hashlib.sha256(
            json.dumps(
                structure_payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _input_signature(source: Path, title: str) -> str:
    source = Path(source)
    payload = {
        "source_sha256": sha256_file(source),
        "title": str(title),
        "contract": CONTRACT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _manifest_path(root: Path) -> Path:
    return Path(root) / ".phoenix_bundle_integrity.json"


def _read_manifest(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _assert_repeat_stability(
    final_root: Path,
    *,
    input_signature: str,
    structure_sha256: str,
) -> None:
    previous = _read_manifest(_manifest_path(final_root))
    if not previous:
        return
    if int(previous.get("contract", 0) or 0) != CONTRACT_VERSION:
        return
    if str(previous.get("input_signature", "")) != input_signature:
        return
    old = str(previous.get("structure_sha256", ""))
    if old and old != structure_sha256:
        raise OutputContractError(
            "同一输入重复生成的成品结构摘要发生变化。"
            "Phoenix已阻止覆盖上一份稳定成品；请先检查生成链是否发生漂移。"
        )


def _write_manifest(root: Path, payload: dict) -> None:
    path = _manifest_path(root)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _begin_directory_publish(staged: Path, final: Path) -> Path | None:
    staged = Path(staged)
    final = Path(final)
    incoming = final.with_name(final.name + ".incoming")
    backup = final.with_name(final.name + ".backup")
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    os.replace(staged, incoming)
    old_exists = final.exists()
    if old_exists:
        os.replace(final, backup)
    try:
        os.replace(incoming, final)
    except Exception:
        if old_exists and backup.exists() and not final.exists():
            try:
                os.replace(backup, final)
            except Exception:
                pass
        raise
    return backup if old_exists else None


def _rollback_directory_publish(final: Path, backup: Path | None) -> None:
    final = Path(final)
    failed = final.with_name(final.name + ".failed")
    shutil.rmtree(failed, ignore_errors=True)
    try:
        if final.exists():
            os.replace(final, failed)
    except Exception:
        pass
    try:
        if backup is not None and backup.exists():
            os.replace(backup, final)
    finally:
        shutil.rmtree(failed, ignore_errors=True)


def _commit_directory_publish(backup: Path | None) -> None:
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def transactional_export_path(exporter, source: Path, *, title: str | None = None):
    """Build all four formats in staging and publish only after full validation."""

    from .rich_export import ExportBundle, MultiFormatExporter

    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    effective_title = title or source.stem
    output_root = Path(exporter.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / f".pxexport-{uuid.uuid4().hex[:10]}"
    shutil.rmtree(stage_root, ignore_errors=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    staged_exporter = MultiFormatExporter(stage_root)
    backup: Path | None = None
    final_root: Path | None = None
    did_publish = False
    try:
        staged_bundle = staged_exporter.export_path(source, title=effective_title)
        staged_report = validate_export_bundle(staged_bundle)
        final_root = output_root / staged_bundle.output_dir.name
        signature = _input_signature(source, str(effective_title))
        _assert_repeat_stability(
            final_root,
            input_signature=signature,
            structure_sha256=str(staged_report["structure_sha256"]),
        )
        _write_manifest(
            staged_bundle.output_dir,
            {
                "contract": CONTRACT_VERSION,
                "input_signature": signature,
                "structure_sha256": staged_report["structure_sha256"],
                "report": staged_report,
            },
        )
        backup = _begin_directory_publish(staged_bundle.output_dir, final_root)
        did_publish = True

        published_bundle = ExportBundle(
            output_dir=final_root,
            markdown=final_root / staged_bundle.markdown.name,
            text=final_root / staged_bundle.text.name,
            docx=final_root / staged_bundle.docx.name,
            pdf=final_root / staged_bundle.pdf.name,
        )
        final_report = validate_export_bundle(published_bundle)
        if final_report["structure_sha256"] != staged_report["structure_sha256"]:
            raise OutputContractError(
                "多格式成品发布后结构摘要变化，疑似文件系统写入/替换异常。"
            )
        manifest = _read_manifest(_manifest_path(final_root))
        if str(manifest.get("structure_sha256", "")) != str(
            final_report["structure_sha256"]
        ):
            raise OutputContractError("多格式成品完整性清单与正式文件不一致。")

        _commit_directory_publish(backup)
        backup = None
        did_publish = False
        return published_bundle
    except Exception:
        if final_root is not None and did_publish:
            _rollback_directory_publish(final_root, backup)
            backup = None
            did_publish = False
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
