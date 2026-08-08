import sys

from PySide6.QtWidgets import QApplication

from config.settings import *
from utils.logger import setup_logger
from ui.main_window import MainWindow


def main():
    logger = setup_logger()
    logger.info("Project Phoenix 启动")

    print("=" * 50)
    print(PROJECT_NAME)
    print("版本：", VERSION)
    print("开发模式：", DEBUG)
    print("支持设备：", SUPPORTED_MODALITIES)
    print("=" * 50)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()