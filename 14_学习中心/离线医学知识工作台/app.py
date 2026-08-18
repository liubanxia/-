from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description="Phoenix 离线医学知识工作台")
    parser.add_argument("--ingest", nargs="*", metavar="PDF", help="导入一个或多个PDF")
    parser.add_argument("--ask", help="只依据PDF知识库回答")
    parser.add_argument("--organize", help="深度整理专题名称")
    parser.add_argument("--instruction", help="深度整理要求")
    parser.add_argument("--resume-task", type=int, help="继续一个未完成的深度整理任务ID")
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
                        f"WARNING={result.warning}",
                        flush=True,
                    )

            if args.build_embeddings:
                count = workbench.retriever.embeddings.build_missing(
                    progress=lambda done, total, msg: print(msg, flush=True)
                )
                print(f"EMBEDDINGS_ADDED={count}")

            if args.ask:
                print(workbench.ask(args.ask).text)

            if args.organize:
                if not args.instruction:
                    raise SystemExit("--organize 必须同时提供 --instruction")
                output, task_id = workbench.organize(args.organize, args.instruction)
                print(f"TASK_ID={task_id}\nOUTPUT={output}")

            if args.resume_task is not None:
                output, task_id = workbench.resume_task(
                    args.resume_task,
                    progress=lambda done, total, msg: print(
                        f"{done}/{total} {msg}",
                        flush=True,
                    ),
                )
                print(f"RESUMED_TASK_ID={task_id}\nOUTPUT={output}")

            if args.status:
                print(json.dumps(workbench.status(), ensure_ascii=False, indent=2))
        finally:
            workbench.close()

        if args.no_gui:
            return 0

    from phoenix_knowledge.gui import run_gui

    return int(run_gui())


if __name__ == "__main__":
    raise SystemExit(main())
