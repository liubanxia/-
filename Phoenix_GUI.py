from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "01_开发源码"

sys.path.insert(0, str(SRC))

try:
    from main_minimal import main
    raise SystemExit(main())

except SystemExit:
    raise

except Exception:
    error = traceback.format_exc()

    log_dir = ROOT / "09_日志"
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / "Phoenix_GUI_启动错误.log").write_text(
        error,
        encoding="utf-8",
    )

    try:
        from PySide6.QtWidgets import (
            QApplication,
            QMessageBox,
        )

        app = QApplication.instance() or QApplication(
            sys.argv
        )

        QMessageBox.critical(
            None,
            "Phoenix 启动失败",
            error[-4000:],
        )
    except Exception:
        pass

    raise
