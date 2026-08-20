from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


_CITATION_RE = re.compile(r"\[S\d+\]")


def _ok(label: str, detail: str = "") -> None:
    print(f"PASS  {label}" + (f" | {detail}" if detail else ""), flush=True)


def _fail(label: str, detail: str) -> None:
    print(f"FAIL  {label} | {detail}", flush=True)


def _make_acceptance_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    texts = (
        "CT demonstrates no pleural effusion and a 12 mm lesion in the right kidney.",
        "MRI demonstrates a small lesion in the left hepatic lobe without restricted diffusion.",
    )
    for text in texts:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(48, 60, 545, 760),
            text,
            fontsize=12,
        )
    doc.save(str(path))
    doc.close()


def _validate_evidence_pages(evidence) -> None:
    import fitz

    checked = set()
    for item in evidence:
        path = Path(item.path)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        key = str(path.resolve())
        if key in checked:
            continue
        checked.add(key)
        doc = fitz.open(str(path))
        try:
            max_page = doc.page_count
        finally:
            doc.close()
        for hit in evidence:
            if Path(hit.path).resolve() == path.resolve():
                if int(hit.page) < 1 or int(hit.page) > max_page:
                    raise RuntimeError(
                        f"引用页码越界：{hit.title} page={hit.page}，实际PDF={max_page}页"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix 正式上线真实平台验收")
    parser.add_argument(
        "--query",
        default="肿瘤影像学中CT有什么作用？",
        help="用于真实资料检索、问答和联合整理的验收问题",
    )
    parser.add_argument(
        "--skip-organize",
        action="store_true",
        help="只在需要快速复测时跳过多资料联合整理",
    )
    args = parser.parse_args()

    os.environ.setdefault("PHOENIX_KNOWLEDGE_ACCELERATOR", "auto")
    os.environ["PHOENIX_KNOWLEDGE_DEEP_QA"] = "1"
    os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = "fast"

    from phoenix_knowledge import MedicalKnowledgeWorkbench
    from phoenix_knowledge.translation_pdf import LAYOUT_TRANSLATED_ONLY
    from phoenix_knowledge.translator import EXPORT_PDF

    failures: list[str] = []
    workbench = MedicalKnowledgeWorkbench()
    try:
        status = workbench.status()
        print("========== Phoenix 真实平台上线验收 ==========", flush=True)
        print(json.dumps({
            "documents": status["documents"],
            "chunks": status["chunks"],
            "semantic_label": status.get("semantic_label"),
            "compute": workbench.llm.compute_status(),
            "translation_backends": status.get("translation_backends"),
            "commercial_release": status.get("commercial_release"),
        }, ensure_ascii=False, indent=2), flush=True)

        try:
            if int(status["documents"]) <= 0 or int(status["chunks"]) <= 0:
                raise RuntimeError("当前资料库为空")
            if not status.get("semantic_ready"):
                raise RuntimeError(str(status.get("semantic_label") or "语义检索未就绪"))
            _ok(
                "资料库与语义索引",
                f"{status['documents']}份 / {status['chunks']}块 / "
                f"{status.get('embedding_vectors', 0)}向量",
            )
        except Exception as exc:
            failures.append(f"资料库与语义索引: {exc}")
            _fail("资料库与语义索引", str(exc))

        hits = []
        try:
            hits = workbench.retriever.search(
                args.query,
                limit=8,
                use_embeddings=True,
            )
            if not hits:
                raise RuntimeError("中文问题未召回任何真实资料")
            _validate_evidence_pages(hits)
            _ok(
                "中文跨语言检索",
                f"HITS={len(hits)} / TOP={hits[0].citation} {hits[0].title} 第{hits[0].page}页",
            )
        except Exception as exc:
            failures.append(f"中文跨语言检索: {exc}")
            _fail("中文跨语言检索", str(exc))

        try:
            answer = workbench.ask(args.query, deep=True)
            if not answer.evidence:
                raise RuntimeError("问答没有证据")
            if not _CITATION_RE.search(answer.text):
                raise RuntimeError("问答输出没有[S编号]引用")
            _ok(
                "真实资料问答",
                f"mode={answer.mode} / evidence={len(answer.evidence)}",
            )
        except Exception as exc:
            failures.append(f"真实资料问答: {exc}")
            _fail("真实资料问答", str(exc))

        try:
            decision = workbench.translator.engine.translate(
                "CT demonstrates no pleural effusion and a 12 mm lesion in the right kidney.",
                "中文",
                smart_level="smart1",
            )
            if not decision.quality.ok:
                raise RuntimeError(
                    "医学翻译质量门未通过：" + "; ".join(decision.quality.reasons)
                )
            translated = decision.text
            if "12" not in translated or "mm" not in translated:
                raise RuntimeError("12 mm 未完整保留")
            _ok(
                "医学翻译与安全校验",
                f"{decision.backend} / score={decision.quality.score:.2f} / {translated}",
            )
        except Exception as exc:
            failures.append(f"医学翻译与安全校验: {exc}")
            _fail("医学翻译与安全校验", str(exc))

        try:
            with tempfile.TemporaryDirectory(prefix="phoenix_release_") as td:
                synthetic = Path(td) / "Phoenix_上线验收.pdf"
                _make_acceptance_pdf(synthetic)
                result = workbench.translate_book(
                    synthetic,
                    start_page=1,
                    target_language="中文",
                    smart_level="smart1",
                    output_layout=LAYOUT_TRANSLATED_ONLY,
                    export_format=EXPORT_PDF,
                    part_pages=50,
                    progress=lambda done, total, msg: print(
                        f"TRANSLATE {done}/{total} {msg}",
                        flush=True,
                    ),
                )
                outputs = tuple(result.output_paths or (result.output_path,))
                if not outputs or not all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in outputs):
                    raise RuntimeError("整本翻译没有生成完整有效PDF成品")
                _ok(
                    "整本PDF翻译与成品导出",
                    f"outputs={len(outputs)} / warning_pages={result.warning_pages}",
                )
        except Exception as exc:
            failures.append(f"整本PDF翻译与成品导出: {exc}")
            _fail("整本PDF翻译与成品导出", str(exc))

        if not args.skip_organize:
            try:
                output, task_id = workbench.organize(
                    "Phoenix正式上线多资料验收",
                    f"只根据全部已导入资料回答并整理：{args.query}。"
                    "要求保留来源编号、原始数字、检查技术和鉴别点；不得补充资料外事实。",
                    candidate_limit=32,
                    batch_size=8,
                    progress=lambda done, total, msg: print(
                        f"ORGANIZE {done}/{total} {msg}",
                        flush=True,
                    ),
                )
                output = Path(output)
                if not output.is_file() or output.stat().st_size <= 0:
                    raise RuntimeError("联合整理正文没有生成")
                text = output.read_text(encoding="utf-8", errors="replace")
                if "当前导入资料中未找到明确依据" not in text and not _CITATION_RE.search(text):
                    raise RuntimeError("联合整理输出没有来源编号")
                bundle = workbench.last_export_bundle
                if bundle is None:
                    raise RuntimeError(
                        "联合整理正文完成，但PDF/DOCX/Markdown/TXT输出包未生成："
                        + str(workbench.last_export_error)
                    )
                missing_outputs = [
                    str(p)
                    for p in bundle.output_paths
                    if not Path(p).is_file() or Path(p).stat().st_size <= 0
                ]
                if missing_outputs:
                    raise RuntimeError("多格式输出缺失：" + ", ".join(missing_outputs))
                _ok(
                    "多资料联合整理与多格式输出",
                    f"task={task_id} / PDF+DOCX+MD+TXT",
                )
            except Exception as exc:
                failures.append(f"多资料联合整理与多格式输出: {exc}")
                _fail("多资料联合整理与多格式输出", str(exc))

    finally:
        workbench.close()

    print("========================================", flush=True)
    if failures:
        print(f"PHOENIX_RELEASE_ACCEPTANCE=FAIL ({len(failures)})", flush=True)
        for item in failures:
            print("- " + item, flush=True)
        return 1

    print("PHOENIX_RELEASE_ACCEPTANCE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
