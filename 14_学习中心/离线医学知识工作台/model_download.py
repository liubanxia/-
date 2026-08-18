from __future__ import annotations

import argparse
import json
from pathlib import Path

from phoenix_knowledge.config import get_paths


MODELS = {
    "embedding": {
        "repo_id": "Qwen/Qwen3-Embedding-0.6B",
        "folder": "Qwen3-Embedding-0.6B",
        "role": "PDF语义检索",
    },
    "reranker": {
        "repo_id": "Qwen/Qwen3-Reranker-0.6B",
        "folder": "Qwen3-Reranker-0.6B",
        "role": "候选证据重排序（预留，后续启用）",
    },
    "generator": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "folder": "Qwen3.5-4B",
        "role": "离线问答与深度整理",
    },
}


def _download_huggingface(repo_id: str, target: Path) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
    )
    return Path(path)


def _download_modelscope(repo_id: str, target: Path) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    try:
        path = snapshot_download(
            repo_id,
            local_dir=str(target),
        )
    except TypeError:
        path = snapshot_download(
            repo_id,
            cache_dir=str(target.parent / "_modelscope_cache"),
        )
        path = Path(path)
        if path.resolve() != target.resolve():
            target.mkdir(parents=True, exist_ok=True)
            (target / "MODELSCOPE_CACHE_PATH.txt").write_text(
                str(path), encoding="utf-8"
            )
    return Path(path)


def download_one(name: str, source: str) -> Path:
    spec = MODELS[name]
    paths = get_paths()
    target = paths.model_root / spec["folder"]
    target.mkdir(parents=True, exist_ok=True)

    if source == "modelscope":
        result = _download_modelscope(spec["repo_id"], target)
    elif source == "huggingface":
        result = _download_huggingface(spec["repo_id"], target)
    else:
        raise ValueError(source)

    manifest = {
        "name": name,
        "repo_id": spec["repo_id"],
        "source": source,
        "role": spec["role"],
        "download_result": str(result),
    }
    (target / "PHOENIX_MODEL.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix 医学知识工作台模型下载")
    parser.add_argument(
        "model",
        choices=[*MODELS.keys(), "all"],
        help="要下载的模型",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="下载源；国内网络优先ModelScope",
    )
    args = parser.parse_args()

    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        spec = MODELS[name]
        print(
            f"DOWNLOAD_START name={name} repo={spec['repo_id']} source={args.source}",
            flush=True,
        )
        target = download_one(name, args.source)
        print(f"DOWNLOAD_DONE name={name} target={target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
