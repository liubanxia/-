import sys

from PySide6.QtWidgets import QApplication

from ui.phoenix_minimal_window import (
    PhoenixMinimalWindow,
)


def main():
    app = QApplication.instance() or QApplication(
        sys.argv
    )

    window = PhoenixMinimalWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
