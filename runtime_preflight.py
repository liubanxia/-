from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CORE_REQUIRED = (
    ("PySide6", "PySide6"),
    ("numpy", "numpy"),
    ("PyMuPDF", "pymupdf"),
    ("pypdf", "pypdf"),
    ("python-docx", "docx"),
    ("Pillow", "PIL"),
    ("cryptography", "cryptography"),
)

AI_REQUIRED = (
    ("sentence-transformers", "sentence_transformers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("sentencepiece", "sentencepiece"),
    ("torch", "torch"),
)


def _missing_group(requirements) -> list[str]:
    missing: list[str] = []
    for package, module in requirements:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    return missing


def _missing_groups() -> tuple[list[str], list[str]]:
    return _missing_group(CORE_REQUIRED), _missing_group(AI_REQUIRED)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _local_wheelhouses(root: Path) -> list[Path]:
    candidates = (
        root / "02_开发环境" / "wheelhouse",
        root / "02_开发环境" / "wheels",
        root / "00_安装包" / "wheelhouse",
        Path(__file__).resolve().parent / "wheelhouse",
    )
    return [path for path in candidates if path.is_dir()]


def _pip_call(
    requirements: Path,
    *,
    wheelhouse: Path | None = None,
) -> int:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--prefer-binary",
    ]
    if wheelhouse is not None:
        command.extend(
            ["--no-index", "--find-links", str(wheelhouse)]
        )
    else:
        command.extend(["--timeout", "12", "--retries", "1"])
    command.extend(["-r", str(requirements)])
    try:
        return int(subprocess.call(command))
    except Exception as exc:
        print(
            f"依赖修复调用失败：{type(exc).__name__}: {exc}",
            flush=True,
        )
        return 1


def _repair(requirements: Path, root: Path) -> int:
    print(
        "Phoenix 检测到运行组件缺失，正在一次性修复运行环境……",
        flush=True,
    )
    for wheelhouse in _local_wheelhouses(root):
        print(f"优先尝试SSD本地依赖包：{wheelhouse}", flush=True)
        if _pip_call(requirements, wheelhouse=wheelhouse) == 0:
            return 0
    print(
        "本地依赖包不足，尝试当前网络的软件源；"
        "离线环境失败后仍会保留基础资料功能。",
        flush=True,
    )
    return _pip_call(requirements)


def _compute_line() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            memory = (
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3)
            )
            return (
                f"本机GPU：{name} ({memory:.1f}GB) · CUDA可用"
            )
        return "本机GPU：未发现可用CUDA，AI任务将自动使用CPU"
    except Exception as exc:
        return (
            f"AI算力组件：torch不可用（{type(exc).__name__}）；"
            "基础资料检索仍可启动"
        )


def _database_quick_check(workbench) -> None:
    """Fail closed on logical SQLite corruption without modifying user data."""
    with workbench.db._lock:
        row = workbench.db._conn.execute(
            "PRAGMA quick_check"
        ).fetchone()
    result = str(row[0] if row else "").strip().lower()
    if result != "ok":
        raise RuntimeError(
            "知识库SQLite完整性检查未通过。Phoenix已停止继续写入，"
            "原始医学资料不会删除；请先恢复数据库或重新建立索引。"
            f" 检查结果：{result or 'unknown'}"
        )


def _database_snapshot(workbench, *, keep: int = 3) -> Path | None:
    """Create at most one verified SQLite backup per day, then rotate safely."""
    source = Path(workbench.paths.database)
    if not source.is_file() or source.stat().st_size <= 0:
        return None

    backup_root = Path(workbench.paths.runtime_root) / "db_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    target = backup_root / f"knowledge_{stamp}.sqlite3"
    if target.is_file() and target.stat().st_size > 0:
        return target

    temp = backup_root / f"knowledge_{stamp}.sqlite3.tmp"
    temp.unlink(missing_ok=True)
    destination = sqlite3.connect(temp)
    try:
        with workbench.db._lock:
            workbench.db._conn.backup(destination)
        row = destination.execute("PRAGMA quick_check").fetchone()
        if str(row[0] if row else "").strip().lower() != "ok":
            raise RuntimeError("新建知识库备份完整性校验失败")
        destination.commit()
    finally:
        destination.close()

    os.replace(temp, target)
    backups = sorted(
        backup_root.glob("knowledge_*.sqlite3"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old in backups[max(1, int(keep)) :]:
        try:
            old.unlink()
        except OSError:
            pass
    return target


def _capability_flags(workbench, status: dict) -> dict[str, bool]:
    """Report each product capability independently.

    Missing semantic dependencies must not make a working local Qwen appear
    unavailable, and missing sentencepiece must not hide a working Qwen
    translation path.
    """

    fast = bool(
        status.get(
            "generator_fast_ready",
            workbench.llm.available("fast"),
        )
    )
    deep = bool(
        status.get(
            "generator_deep_ready",
            workbench.llm.available("deep"),
        )
    )
    return {
        "semantic": bool(status.get("semantic_ready")),
        "smart1": fast,
        "smart2": deep,
        "translation": bool(status.get("translation_backends") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phoenix 医学知识工作台首次启动自检"
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="缺少运行组件时一次性修复",
    )
    args = parser.parse_args()

    root = _project_root()
    os.environ.setdefault("PHOENIX_PROJECT_ROOT", str(root))
    os.environ.setdefault(
        "PHOENIX_KNOWLEDGE_ACCELERATOR",
        "auto",
    )

    print("========== Phoenix 启动自检 ==========", flush=True)
    print(f"工程：{root}", flush=True)
    print(
        f"Python：{sys.version.split()[0]} · {sys.executable}",
        flush=True,
    )
    print(_compute_line(), flush=True)

    core_missing, ai_missing = _missing_groups()
    if (core_missing or ai_missing) and args.repair:
        requirements = Path(__file__).with_name(
            "requirements-runtime.txt"
        )
        if not requirements.is_file():
            print(
                f"缺少运行依赖清单：{requirements}",
                flush=True,
            )
            return 2
        _repair(requirements, root)
        importlib.invalidate_caches()
        core_missing, ai_missing = _missing_groups()

    if core_missing:
        print(
            "基础运行组件缺失：" + ", ".join(core_missing),
            flush=True,
        )
        print(
            "Phoenix 无法安全打开GUI。请补齐上述基础组件后重试。",
            flush=True,
        )
        return 2

    if ai_missing:
        print(
            "AI增强组件仍缺失：" + ", ".join(ai_missing),
            flush=True,
        )
        print(
            "Phoenix 将继续启动，并按能力分别判断：缺语义组件只影响语义检索，"
            "缺生成组件只影响本地智能问答，缺翻译组件只影响对应翻译后端。",
            flush=True,
        )

    try:
        from phoenix_knowledge import MedicalKnowledgeWorkbench

        workbench = MedicalKnowledgeWorkbench()
        try:
            _database_quick_check(workbench)
            backup = _database_snapshot(workbench)
            if backup is not None:
                print(
                    f"知识库：完整性检查通过 · 安全快照 {backup.name}",
                    flush=True,
                )
            else:
                print("知识库：完整性检查通过", flush=True)

            status = workbench.status()
            capability = _capability_flags(workbench, status)
            print(
                f"资料库：{status['documents']} 份 · "
                f"知识块 {status['chunks']} · "
                f"{status.get('semantic_label', '语义状态未知')}",
                flush=True,
            )
            print(
                f"智能1：{'READY' if capability['smart1'] else '未就绪'} · "
                f"智能2：{'READY' if capability['smart2'] else '未就绪'}",
                flush=True,
            )
            backends = status.get("translation_backends") or []
            print(
                "医学翻译："
                + (
                    "READY · " + ", ".join(backends)
                    if capability["translation"]
                    else "未就绪"
                ),
                flush=True,
            )
        finally:
            workbench.close()
    except Exception as exc:
        print(
            f"工作台核心自检失败：{type(exc).__name__}: {exc}",
            flush=True,
        )
        return 3

    print(
        "自检完成，正在进入 Phoenix 医学知识工作台。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
