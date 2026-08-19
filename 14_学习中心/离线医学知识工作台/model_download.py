from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from phoenix_knowledge.config import get_paths, resolve_model_dir


DEFAULT_HF_MIRRORS = ["https://hf-mirror.com"]

MODELS = {
    "embedding": {
        "repo_id": "Qwen/Qwen3-Embedding-0.6B",
        "modelscope_id": "Qwen/Qwen3-Embedding-0.6B",
        "folder": "Qwen3-Embedding-0.6B",
        "role": "PDF语义检索",
        "required_any": [
            ["config.json"],
            ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"],
        ],
    },
    "reranker": {
        "repo_id": "Qwen/Qwen3-Reranker-0.6B",
        "modelscope_id": "Qwen/Qwen3-Reranker-0.6B",
        "folder": "Qwen3-Reranker-0.6B",
        "role": "候选证据重排序（预留，后续启用）",
        "required_any": [
            ["config.json"],
            ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"],
        ],
    },
    "generator": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "modelscope_id": "Qwen/Qwen3.5-4B",
        "folder": "Qwen3.5-4B",
        "role": "离线问答、深度整理、医学翻译最终兜底",
        "required_any": [
            ["config.json"],
            ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"],
        ],
    },
    "translation_fast": {
        "repo_id": "Helsinki-NLP/opus-mt-en-zh",
        "modelscope_id": "Helsinki-NLP/opus-mt-en-zh",
        "folder": "opus-mt-en-zh",
        "role": "快速英译中专用模型；整本翻译第一模型",
        "ignore_patterns": ["*.h5", "*.msgpack", "*.ot"],
        "required_any": [
            ["config.json"],
            ["source.spm"],
            ["target.spm"],
            ["model.safetensors", "pytorch_model.bin"],
        ],
    },
    "translation_backup": {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "modelscope_id": "facebook/nllb-200-distilled-600M",
        "folder": "NLLB-200-distilled-600M",
        "role": "整本翻译第二兜底模型；研究/非商业用途",
        "ignore_patterns": ["*.h5", "*.msgpack", "*.ot", "onnx/*"],
        "required_any": [
            ["config.json"],
            ["sentencepiece.bpe.model"],
            ["model.safetensors", "pytorch_model.bin"],
        ],
    },
}

GROUPS = {
    "translation": ["translation_fast", "translation_backup", "generator"],
    "translation_light": ["translation_fast"],
    "knowledge": ["embedding", "reranker", "generator"],
    "hospital_recommended": ["translation_fast", "embedding", "generator"],
    "all": list(MODELS),
}


def _mirror_list() -> list[str]:
    raw = os.environ.get("PHOENIX_HF_MIRRORS", "").strip()
    if not raw:
        return list(DEFAULT_HF_MIRRORS)
    mirrors = []
    for item in raw.replace(",", ";").split(";"):
        item = item.strip().rstrip("/")
        if item and item not in mirrors:
            mirrors.append(item)
    return mirrors or list(DEFAULT_HF_MIRRORS)


def build_routes(source: str, mirrors: list[str] | None = None) -> list[tuple[str, str | None]]:
    mirrors = mirrors if mirrors is not None else _mirror_list()
    if source == "modelscope":
        return [("modelscope", None)]
    if source == "huggingface":
        return [("huggingface", "https://huggingface.co")]
    if source == "hf-mirror":
        return [("hf-mirror", endpoint) for endpoint in mirrors]
    if source != "auto":
        raise ValueError(source)

    routes: list[tuple[str, str | None]] = [("modelscope", None)]
    routes.extend(("hf-mirror", endpoint) for endpoint in mirrors)
    routes.append(("huggingface", "https://huggingface.co"))
    return routes


def _download_huggingface(
    repo_id: str,
    target: Path,
    *,
    endpoint: str,
    ignore_patterns=None,
) -> Path:
    # HfApi(endpoint=...) avoids relying on process-global HF_ENDPOINT and makes
    # it safe to try multiple routes sequentially in one hospital-side process.
    from huggingface_hub import HfApi

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    if endpoint != "https://huggingface.co":
        # Community mirrors may not support the Xet transport consistently.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    api = HfApi(endpoint=endpoint)
    path = api.snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=ignore_patterns,
    )
    return Path(path)


def _download_modelscope(repo_id: str, target: Path) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    try:
        path = snapshot_download(repo_id, local_dir=str(target))
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


def _resolved_model_dir(target: Path) -> Path:
    pointer = target / "MODELSCOPE_CACHE_PATH.txt"
    if pointer.is_file():
        try:
            path = Path(pointer.read_text(encoding="utf-8").strip())
            if path.is_dir():
                return path
        except OSError:
            pass
    return target


def validate_download(name: str, target: Path) -> tuple[bool, list[str]]:
    spec = MODELS[name]
    root = _resolved_model_dir(target)
    missing: list[str] = []

    if not root.is_dir():
        return False, [f"模型目录不存在: {root}"]

    for alternatives in spec.get("required_any", []):
        if not any((root / filename).is_file() and (root / filename).stat().st_size > 0 for filename in alternatives):
            missing.append(" / ".join(alternatives))

    # Detect common interrupted-download leftovers. They can remain for resume,
    # but the model must not be reported READY while only temporary files exist.
    model_payloads = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".bin", ".safetensors", ".model", ".spm"}
        and p.stat().st_size > 0
    ]
    if not model_payloads:
        missing.append("未发现有效模型权重/分词器文件")

    return not missing, missing


def _write_route_status(target: Path, payload: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "PHOENIX_DOWNLOAD_STATUS.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def download_one(
    name: str,
    source: str = "auto",
    *,
    retries: int = 2,
    retry_delay: float = 3.0,
) -> Path:
    spec = MODELS[name]
    paths = get_paths()
    target = paths.model_root / spec["folder"]
    target.mkdir(parents=True, exist_ok=True)

    already_ok, _ = validate_download(name, target)
    if already_ok:
        print(f"MODEL_ALREADY_READY name={name} target={_resolved_model_dir(target)}", flush=True)
        return target

    attempts: list[dict] = []
    last_error: Exception | None = None

    for route_name, endpoint in build_routes(source):
        for attempt_index in range(1, max(1, retries) + 1):
            started = time.time()
            print(
                f"ROUTE_START name={name} route={route_name} "
                f"endpoint={endpoint or '-'} attempt={attempt_index}/{max(1, retries)}",
                flush=True,
            )
            try:
                if route_name == "modelscope":
                    result = _download_modelscope(
                        spec.get("modelscope_id", spec["repo_id"]), target
                    )
                else:
                    result = _download_huggingface(
                        spec["repo_id"],
                        target,
                        endpoint=str(endpoint),
                        ignore_patterns=spec.get("ignore_patterns"),
                    )

                ok, missing = validate_download(name, target)
                if not ok:
                    raise RuntimeError("下载后完整性检查失败: " + "; ".join(missing))

                record = {
                    "route": route_name,
                    "endpoint": endpoint,
                    "attempt": attempt_index,
                    "ok": True,
                    "seconds": round(time.time() - started, 2),
                    "result": str(result),
                }
                attempts.append(record)
                manifest = {
                    "name": name,
                    "repo_id": spec["repo_id"],
                    "source": route_name,
                    "endpoint": endpoint,
                    "role": spec["role"],
                    "download_result": str(result),
                    "validated": True,
                    "attempts": attempts,
                }
                (target / "PHOENIX_MODEL.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _write_route_status(target, {"status": "completed", **manifest})
                print(
                    f"ROUTE_SUCCESS name={name} route={route_name} "
                    f"target={_resolved_model_dir(target)}",
                    flush=True,
                )
                return target
            except Exception as exc:
                last_error = exc
                record = {
                    "route": route_name,
                    "endpoint": endpoint,
                    "attempt": attempt_index,
                    "ok": False,
                    "seconds": round(time.time() - started, 2),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                attempts.append(record)
                _write_route_status(
                    target,
                    {
                        "status": "retrying",
                        "name": name,
                        "repo_id": spec["repo_id"],
                        "attempts": attempts,
                    },
                )
                print(
                    f"ROUTE_FAILED name={name} route={route_name} "
                    f"attempt={attempt_index} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt_index < max(1, retries):
                    time.sleep(max(0.0, retry_delay))

    _write_route_status(
        target,
        {
            "status": "failed",
            "name": name,
            "repo_id": spec["repo_id"],
            "attempts": attempts,
        },
    )
    raise RuntimeError(
        f"所有下载线路均失败: {name}. 最后错误: {type(last_error).__name__ if last_error else 'Unknown'}: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix 医学知识工作台多线路模型下载")
    parser.add_argument(
        "model",
        choices=[*MODELS.keys(), *GROUPS.keys()],
        help="要下载的单模型或模型组",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "modelscope", "hf-mirror", "huggingface"],
        default="auto",
        help="auto=ModelScope→国内HF镜像→Hugging Face官方；医院电脑推荐auto",
    )
    parser.add_argument("--retries", type=int, default=2, help="每条线路重试次数")
    parser.add_argument(
        "--retry-delay", type=float, default=3.0, help="同一线路重试间隔秒数"
    )
    parser.add_argument("--list-routes", action="store_true", help="只显示自动线路顺序")
    args = parser.parse_args()

    if args.list_routes:
        for index, (route, endpoint) in enumerate(build_routes(args.source), start=1):
            print(f"{index}. {route}: {endpoint or 'ModelScope'}")
        return 0

    names = GROUPS.get(args.model, [args.model])
    failures: list[str] = []
    for name in names:
        spec = MODELS[name]
        print(
            f"DOWNLOAD_START name={name} repo={spec['repo_id']} source={args.source}",
            flush=True,
        )
        try:
            target = download_one(
                name,
                args.source,
                retries=max(1, args.retries),
                retry_delay=max(0.0, args.retry_delay),
            )
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"DOWNLOAD_FAILED {failures[-1]}", flush=True)
            continue
        print(f"DOWNLOAD_DONE name={name} target={target}", flush=True)

    if failures:
        print("\n以下模型未完成：", flush=True)
        for item in failures:
            print(f"- {item}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
