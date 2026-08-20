from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


REQUIRED = (
    ("PySide6", "PySide6"),
    ("numpy", "numpy"),
    ("PyMuPDF", "pymupdf"),
    ("pypdf", "pypdf"),
    ("python-docx", "docx"),
    ("Pillow", "PIL"),
    ("sentence-transformers", "sentence_transformers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("sentencepiece", "sentencepiece"),
    ("cryptography", "cryptography"),
)


def _missing() -> list[str]:
    missing = []
    for package, module in REQUIRED:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    return missing


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repair(requirements: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--prefer-binary",
        "-r",
        str(requirements),
    ]
    print("Phoenix 检测到运行组件缺失，正在一次性修复运行环境……", flush=True)
    try:
        return int(subprocess.call(command))
    except Exception as exc:
        print(f"自动修复失败：{type(exc).__name__}: {exc}", flush=True)
        return 1


def _compute_line() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"本机GPU：{name} ({memory:.1f}GB) · CUDA可用"
        return "本机GPU：未发现可用CUDA，工作台将自动使用CPU"
    except Exception as exc:
        return f"本机算力：Torch检测失败（{type(exc).__name__}）"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix 医学知识工作台首次启动自检")
    parser.add_argument("--repair", action="store_true", help="缺少运行组件时一次性安装")
    args = parser.parse_args()

    root = _project_root()
    os.environ.setdefault("PHOENIX_PROJECT_ROOT", str(root))
    os.environ.setdefault("PHOENIX_KNOWLEDGE_ACCELERATOR", "auto")

    print("========== Phoenix 启动自检 ==========", flush=True)
    print(f"工程：{root}", flush=True)
    print(f"Python：{sys.version.split()[0]} · {sys.executable}", flush=True)
    print(_compute_line(), flush=True)

    missing = _missing()
    if missing and args.repair:
        requirements = Path(__file__).with_name("requirements-runtime.txt")
        if not requirements.is_file():
            print(f"缺少运行依赖清单：{requirements}", flush=True)
            return 2
        if _repair(requirements) != 0:
            return 2
        importlib.invalidate_caches()
        missing = _missing()

    if missing:
        print("运行组件缺失：" + ", ".join(missing), flush=True)
        print("请联网后重新双击启动器；启动器会一次性修复，不需要逐个安装。", flush=True)
        return 2

    try:
        from phoenix_knowledge import MedicalKnowledgeWorkbench
        workbench = MedicalKnowledgeWorkbench()
        try:
            status = workbench.status()
            print(
                f"资料库：{status['documents']} 份 · "
                f"知识块 {status['chunks']} · "
                f"{status.get('semantic_label', '语义状态未知')}",
                flush=True,
            )
            print(
                f"智能1：{'READY' if workbench.llm.available('fast') else '未就绪'} · "
                f"智能2：{'READY' if workbench.llm.available('deep') else '未就绪'}",
                flush=True,
            )
            backends = status.get("translation_backends") or []
            print(
                "医学翻译：" + ("READY · " + ", ".join(backends) if backends else "未就绪"),
                flush=True,
            )
        finally:
            workbench.close()
    except Exception as exc:
        print(f"工作台核心自检失败：{type(exc).__name__}: {exc}", flush=True)
        return 3

    print("自检完成，正在进入 Phoenix 医学知识工作台。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
