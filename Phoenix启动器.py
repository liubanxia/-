from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime


def project_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = project_root()

PYTHON = (
    ROOT
    / "04_AI模型"
    / "工程工作区"
    / "phoenix_distill_env"
    / "Scripts"
    / "python.exe"
)

TARGET = ROOT / "01_开发源码" / "phoenix_minimal.py"

LOG_DIR = ROOT / "09_日志"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG = LOG_DIR / "Phoenix_启动.log"


def fail(text):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"{text}\n"
        )

    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            text,
            "Phoenix 启动失败",
            0x10,
        )
    except Exception:
        pass


if not PYTHON.exists():
    fail(f"找不到 Python：\n{PYTHON}")
    raise SystemExit(1)

if not TARGET.exists():
    fail(f"找不到 Phoenix 主程序：\n{TARGET}")
    raise SystemExit(1)


env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "01_开发源码")
env["PHOENIX_ROOT"] = str(ROOT)

with LOG.open("a", encoding="utf-8") as log:
    log.write(
        f"\n===== Phoenix 启动 "
        f"{datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
    )

    subprocess.Popen(
        [str(PYTHON), str(TARGET)],
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=log,
    )
