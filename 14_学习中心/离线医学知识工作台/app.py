from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description="Phoenix 离线医学知识工作台")
    parser.add_argument("--ingest", nargs="*", metavar="PDF", help="导入一个或多个PDF")
    parser.add_argument("--ask", help="只依据PDF知识库回答")
    parser.add_argument("--deep-qa", action="store_true", help="问答时启用Qwen深度归纳；默认使用快速证据模式")
    parser.add_argument("--organize", help="多书深度整理专题名称")
    parser.add_argument("--instruction", help="整理要求")
    parser.add_argument("--resume-task", type=int, help="继续一个未完成的多书整理任务ID")
    parser.add_argument("--translate-book", metavar="PDF", help="整本PDF离线翻译")
    parser.add_argument("--start-page", type=int, default=1, help="整本翻译从第几页开始，默认1")
    parser.add_argument("--target-language", default="中文", help="翻译目标语言，默认中文")
    parser.add_argument("--retry-warning-pages", action="store_true", help="整本翻译时重新处理带警告的页面")
    parser.add_argument("--organize-txt", metavar="TXT", help="整理一个TXT/MD医学笔记")
    parser.add_argument("--note-title", help="TXT整理后的笔记标题")
    parser.add_argument("--build-embeddings", action="store_true", help="为已导入知识块建立本地向量索引")
    parser.add_argument("--status", action="store_true", help="显示知识库状态")
    parser.add_argument("--no-gui", action="store_true", help="完成命令行任务后不启动窗口")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    has_cli_action = bool(
        args.ingest
        or args.ask
        or args.organize
        or args.resume_task is not None
        or args.translate_book
        or args.organize_txt
        or args.build_embeddings
        or args.status
    )

    if has_cli_action:
        from phoenix_knowledge import MedicalKnowledgeWorkbench

        workbench = MedicalKnowledgeWorkbench()
        try:
            if args.ingest:
                for filename in args.ingest:
                    result = workbench.ingest(
                        Path(filename),
                        progress=lambda done, total, msg: print(msg, flush=True),
                    )
                    print(
                        f"INGESTED={result.copied_to_library} "
                        f"PAGES={result.pages_indexed}/{result.pages_total} "
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
                answer = workbench.ask(args.ask, deep=bool(args.deep_qa))
                print(f"ANSWER_MODE={answer.mode}")
                print(answer.text)

            if args.organize:
                if not args.instruction:
                    raise SystemExit("--organize 必须同时提供 --instruction")
                output, task_id = workbench.organize(
                    args.organize,
                    args.instruction,
                    progress=lambda done, total, msg: print(
                        f"{done}/{total} {msg}", flush=True
                    ),
                )
                print(f"TASK_ID={task_id}\nOUTPUT={output}")

            if args.resume_task is not None:
                output, task_id = workbench.resume_task(
                    args.resume_task,
                    progress=lambda done, total, msg: print(
                        f"{done}/{total} {msg}", flush=True
                    ),
                )
                print(f"RESUMED_TASK_ID={task_id}\nOUTPUT={output}")

            if args.translate_book:
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
                    f"BACKENDS={','.join(result.available_backends)}"
                )

            if args.organize_txt:
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
                print(json.dumps(workbench.status(), ensure_ascii=False, indent=2))
        finally:
            workbench.close()

        if args.no_gui:
            return 0

    from phoenix_knowledge import gui as gui_module
    try:
        from phoenix_knowledge.gui_enhancements import install as install_gui_enhancements
        install_gui_enhancements(gui_module)
    except Exception as exc:
        print(f"GUI_ENHANCEMENT_WARNING={type(exc).__name__}: {exc}", flush=True)
    return int(gui_module.run_gui())


if __name__ == "__main__":
    raise SystemExit(main())
