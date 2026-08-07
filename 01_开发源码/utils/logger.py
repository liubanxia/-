import logging
from pathlib import Path
from datetime import datetime


def setup_logger() -> logging.Logger:
    """
    初始化 Project Phoenix 日志系统。
    """

    project_root = Path(__file__).resolve().parents[2]

    log_dir = project_root / "09_日志"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{datetime.now():%Y-%m-%d}.log"

    logger = logging.getLogger("ProjectPhoenix")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger