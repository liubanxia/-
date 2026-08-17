try:
    from PySide6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QTextEdit,
        QPushButton,
        QApplication,
    )
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QTextEdit,
            QPushButton,
            QApplication,
        )
    except ImportError:
        from PyQt5.QtWidgets import (
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QTextEdit,
            QPushButton,
            QApplication,
        )


class AIReportWindow(QWidget):

    def __init__(self, controller=None, parent=None):
        super().__init__(parent)

        self.controller = controller

        self.setWindowTitle("Phoenix AI报告")
        self.resize(720, 600)

        root = QVBoxLayout(self)

        title = QLabel("AI报告")
        root.addWidget(title)

        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setPlaceholderText(
            "AI报告将在这里显示"
        )

        root.addWidget(self.editor, 1)

        buttons = QHBoxLayout()

        self.copy_btn = QPushButton("复制报告")
        self.close_btn = QPushButton("关闭")

        buttons.addWidget(self.copy_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.close_btn)

        root.addLayout(buttons)

        self.copy_btn.clicked.connect(
            self.copy_report
        )

        self.close_btn.clicked.connect(
            self.close
        )

    def set_controller(self, controller):
        self.controller = controller

    def set_report(self, text):
        self.editor.setPlainText(
            str(text or "")
        )

    def report_text(self):
        return self.editor.toPlainText()

    def copy_report(self):
        QApplication.clipboard().setText(
            self.report_text()
        )
