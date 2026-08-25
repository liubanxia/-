from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description="Phoenix 离线医学知识工作台")
    parser.add_argument(
        "--ingest",
        nargs="*",
        metavar="FILE",
        help="导入一个或多个医学资料：PDF/PPT/PPTX/DOCX/TXT/MD",
    )
    parser.add_argument("--ask", help="只依据已导入医学资料知识库回答")
    parser.add_argument(
        "--deep-qa",
        action="store_true",
        help="问答时启用智能归纳；默认仍是快速证据模式",
    )
    parser.add_argument(
        "--deep-4b",
        action="store_true",
        help="问答时强制智能2深度质量模式（最慢）",
    )
    parser.add_argument("--organize", help="多资料深度整理专题名称")
    parser.add_argument("--instruction", help="整理要求")
    parser.add_argument("--resume-task", type=int, help="继续一个未完成的多资料整理任务ID")
    parser.add_argument(
        "--translate-book",
        metavar="DOCUMENT",
        help="同格式医学翻译：PDF→PDF、PPTX→PPTX、DOCX→DOCX",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="从第几页/幻灯片/论文单元开始，默认1",
    )
    parser.add_argument("--target-language", default="中文", help="翻译目标语言，默认中文")
    parser.add_argument(
        "--retry-warning-pages",
        action="store_true",
        help="重新处理带警告的页面/幻灯片/论文单元",
    )
    parser.add_argument("--organize-txt", metavar="TXT", help="整理一个TXT/MD医学笔记")
    parser.add_argument("--note-title", help="TXT整理后的笔记标题")
    parser.add_argument("--build-embeddings", action="store_true", help="为已导入知识块建立本地向量索引")
    parser.add_argument("--status", action="store_true", help="显示知识库状态")
    parser.add_argument("--machine-code", action="store_true", help="显示正式版离线授权机器码")
    parser.add_argument("--activate", metavar="CODE", help="写入并验证离线激活码")
    parser.add_argument("--license-status", action="store_true", help="显示产品授权状态")
    parser.add_argument(
        "--compute",
        choices=["auto", "cpu", "cuda", "deepspeed", "remote"],
        help="算力来源：自动/CPU/CUDA/DeepSpeed/外接GPU API",
    )
    parser.add_argument(
        "--gpu-url",
        help="外接OpenAI兼容GPU/API地址，例如 https://api.deepseek.com 或局域网服务",
    )
    parser.add_argument("--gpu-model-fast", help="外接服务快速模型名")
    parser.add_argument("--gpu-model-deep", help="外接服务质量模型名")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="本次运行明确允许把当前处理文本发送到外接GPU/API",
    )
    parser.add_argument("--no-gui", action="store_true", help="完成命令行任务后不启动窗口")
    return parser


def _apply_compute_cli(args) -> None:
    if args.compute:
        os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = args.compute
    if args.gpu_url:
        os.environ["PHOENIX_KNOWLEDGE_REMOTE_URL"] = args.gpu_url
    if args.gpu_model_fast:
        os.environ["PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST"] = args.gpu_model_fast
    if args.gpu_model_deep:
        os.environ["PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP"] = args.gpu_model_deep
    if args.allow_remote:
        os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "1"


def _non_license_cli_action(args) -> bool:
    return bool(
        args.ingest
        or args.ask
        or args.organize
        or args.resume_task is not None
        or args.translate_book
        or args.organize_txt
        or args.build_embeddings
        or args.status
    )


def _license_cli_action(args) -> bool:
    return bool(args.machine_code or args.activate or args.license_status)


def _print_export_bundle(workbench) -> None:
    bundle = getattr(workbench, "last_export_bundle", None)
    if bundle is None:
        return
    print("ORGANIZED_FORMATS=" + ",".join(str(path) for path in bundle.output_paths))


def main() -> int:
    args = _build_parser().parse_args()
    _apply_compute_cli(args)
    non_license_action = _non_license_cli_action(args)
    _license_cli_action(args)

    from phoenix_knowledge.config import get_paths
    from phoenix_knowledge.licensing import LicenseManager

    paths = get_paths()
    license_manager = LicenseManager(paths.project_root)

    if args.machine_code:
        print(f"MACHINE_CODE={license_manager.machine_code}")

    if args.activate:
        try:
            activated = license_manager.activate(args.activate)
        except Exception as exc:
            print(f"ACTIVATION_FAILED={type(exc).__name__}: {exc}")
            return 21
        print("ACTIVATION=SUCCESS")
        print(json.dumps(activated.as_dict(), ensure_ascii=False, indent=2))

    if args.license_status:
        print(json.dumps(license_manager.status().as_dict(), ensure_ascii=False, indent=2))

    if non_license_action and license_manager.product_mode:
        current = license_manager.status()
        if not current.valid:
            print("PRODUCT_NOT_ACTIVATED=1")
            print(f"MACHINE_CODE={current.machine_code}")
            print(f"LICENSE_MESSAGE={current.message}")
            return 23

    # Production runtime installation is an explicit application-boundary step.
    # License-only headless commands remain lightweight and do not install the
    # translation/workbench monkey-patch stack.
    runtime_required = bool(non_license_action or not args.no_gui)
    if runtime_required:
        from phoenix_knowledge import bootstrap_runtime

        try:
            bootstrap_runtime()
        except Exception as exc:
            print(
                f"RUNTIME_BOOTSTRAP_FATAL={type(exc).__name__}: {exc}",
                flush=True,
            )
            return 30

    if non_license_action:
        from phoenix_knowledge.workbench import MedicalKnowledgeWorkbench

        workbench = MedicalKnowledgeWorkbench(paths)
        try:
            if args.ingest:
                for filename in args.ingest:
                    result = workbench.ingest(
                        Path(filename),
                        progress=lambda done, total, msg: print(msg, flush=True),
                    )
                    print(
                        f"INGESTED={result.copied_to_library} "
                        f"UNITS={result.pages_indexed}/{result.pages_total} "
                        f"IMAGES={result.image_count} "
                        f"WARNING={result.warning}",
                        flush=True,
                    )

            if args.build_embeddings:
                count = workbench.retriever.embeddings.build_missing(
                    progress=lambda done, total, msg: print(msg, flush=True)
                )
                print(f"EMBEDDINGS_ADDED={count}")

            if args.ask:
                if args.deep_4b:
                    os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = "deep"
                else:
                    os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = "fast"
                answer = workbench.ask(
                    args.ask,
                    deep=bool(args.deep_qa or args.deep_4b),
                )
                print(f"ANSWER_MODE={answer.mode}")
                print(f"GENERATOR={workbench.llm.active_model_name()}")
                print(f"COMPUTE={workbench.llm.compute.status().label()}")
                print(answer.text)

            if args.organize:
                if not args.instruction:
                    raise SystemExit("--organize 必须同时提供 --instruction")
                os.environ.setdefault(
                    "PHOENIX_KNOWLEDGE_LLM_PROFILE",
                    "fast",
                )
                output, task_id = workbench.organize(
                    args.organize,
                    args.instruction,
                    progress=lambda done, total, msg: print(
                        f"{done}/{total} {msg}", flush=True
                    ),
                )
                print(f"TASK_ID={task_id}\nOUTPUT={output}")
                _print_export_bundle(workbench)

            if args.resume_task is not None:
                os.environ.setdefault("PHOENIX_KNOWLEDGE_LLM_PROFILE", "fast")
                output, task_id = workbench.resume_task(
                    args.resume_task,
                    progress=lambda done, total, msg: print(
                        f"{done}/{total} {msg}", flush=True
                    ),
                )
                print(f"RESUMED_TASK_ID={task_id}\nOUTPUT={output}")
                _print_export_bundle(workbench)

            if args.translate_book:
                os.environ.setdefault(
                    "PHOENIX_KNOWLEDGE_LLM_PROFILE",
                    "translation",
                )
                result = workbench.translate_book(
                    Path(args.translate_book),
                    start_page=max(1, int(args.start_page)),
                    target_language=args.target_language,
                    retry_warning_pages=bool(args.retry_warning_pages),
                    progress=lambda done, total, msg: print(
                        f"{done}/{total} {msg}", flush=True
                    ),
                )
                print(
                    f"TRANSLATION_OUTPUT={result.output_path}\n"
                    f"TRANSLATION_FORMATS={','.join(str(x) for x in result.output_paths)}\n"
                    f"IMAGES={result.image_count}\n"
                    f"WARNING_PAGES={result.warning_pages}\n"
                    f"BACKENDS={','.join(result.available_backends)}\n"
                    f"COMPUTE={workbench.llm.compute.status().label()}"
                )

            if args.organize_txt:
                os.environ.setdefault("PHOENIX_KNOWLEDGE_LLM_PROFILE", "fast")
                result = workbench.organize_txt_file(
                    Path(args.organize_txt),
                    title=args.note_title,
                    instruction=args.instruction or "",
                    progress=lambda done, total, msg: print(
                        f"{done}/{total} {msg}", flush=True
                    ),
                )
                print(
                    f"NOTES_OUTPUT={result.output_path}\n"
                    f"MODE={result.mode}\nCHUNKS={result.chunks}"
                )

            if args.status:
                payload = workbench.status()
                payload["compute"] = workbench.llm.compute_status()
                print(json.dumps(payload, ensure_ascii=False, indent=2))
        finally:
            workbench.close()

    if args.no_gui:
        return 0

    if license_manager.product_mode:
        from phoenix_knowledge.activation_ui import ensure_gui_activation

        if not ensure_gui_activation(paths.project_root):
            return 23

    # Runtime is already fully installed above. Import GUI directly instead of
    # depending on package-level attribute import semantics.
    import phoenix_knowledge.gui as gui_module
    from phoenix_knowledge.gui_bootstrap import install_gui_stack

    try:
        install_gui_stack(
            gui_module,
            strict=True,
            reporter=lambda message: print(message, flush=True),
        )
    except Exception as exc:
        print(
            f"GUI_BOOTSTRAP_FATAL={type(exc).__name__}: {exc}",
            flush=True,
        )
        return 31

    return int(gui_module.run_gui())


if __name__ == "__main__":
    raise SystemExit(main())
