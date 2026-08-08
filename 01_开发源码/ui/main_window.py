from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
)


class MainWindow(QMainWindow):
    """Project Phoenix 医学影像工作站主窗口。"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Project Phoenix - 医学影像智能工作站")
        self.resize(1200, 800)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        title_label = QLabel("Project Phoenix")
        subtitle_label = QLabel("医学影像智能工作站")

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        self.setCentralWidget(central_widget)