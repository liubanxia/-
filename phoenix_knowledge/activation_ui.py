from __future__ import annotations

from pathlib import Path

from .licensing import LicenseManager


def ensure_gui_activation(project_root: Path) -> bool:
    """Block release GUI startup until a valid offline activation is present."""

    manager = LicenseManager(project_root)
    if not manager.product_mode:
        return True

    status = manager.status()
    if status.valid:
        return True

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
    )

    app = QApplication.instance() or QApplication([])

    class ActivationDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Phoenix 产品激活")
            self.setMinimumWidth(680)
            self.setModal(True)

            layout = QVBoxLayout(self)
            title = QLabel("Phoenix 医学知识工作台 · 正式授权")
            title.setStyleSheet("font-size: 20px; font-weight: 600;")
            layout.addWidget(title)

            message = QLabel(
                "当前为正式产品模式。首次使用必须输入与你的机器码匹配的离线激活码。"
                "医院电脑不需要联网。"
            )
            message.setWordWrap(True)
            layout.addWidget(message)

            layout.addWidget(QLabel("机器码"))
            machine_row = QHBoxLayout()
            self.machine_edit = QLineEdit(manager.machine_code)
            self.machine_edit.setReadOnly(True)
            copy_button = QPushButton("复制机器码")
            machine_row.addWidget(self.machine_edit, 1)
            machine_row.addWidget(copy_button)
            layout.addLayout(machine_row)

            layout.addWidget(QLabel("激活码"))
            self.code_edit = QTextEdit()
            self.code_edit.setPlaceholderText("粘贴 PHX1. 开头的离线激活码")
            self.code_edit.setMaximumHeight(145)
            layout.addWidget(self.code_edit)

            self.status_label = QLabel(status.message)
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)

            buttons = QHBoxLayout()
            buttons.addStretch(1)
            exit_button = QPushButton("退出")
            activate_button = QPushButton("激活并进入")
            activate_button.setDefault(True)
            buttons.addWidget(exit_button)
            buttons.addWidget(activate_button)
            layout.addLayout(buttons)

            copy_button.clicked.connect(self._copy_machine)
            exit_button.clicked.connect(self.reject)
            activate_button.clicked.connect(self._activate)

        def _copy_machine(self):
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(manager.machine_code)
            self.status_label.setText("机器码已复制。")

        def _activate(self):
            code = self.code_edit.toPlainText().strip()
            if not code:
                self.status_label.setText("请输入激活码。")
                return
            try:
                activated = manager.activate(code)
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Phoenix 激活失败",
                    f"{type(exc).__name__}: {exc}",
                )
                return

            QMessageBox.information(
                self,
                "Phoenix 已激活",
                f"授权编号：{activated.license_id}\n"
                f"版本：{activated.edition}\n"
                f"客户：{activated.customer or '未填写'}\n"
                f"有效期：{activated.expires_at or '永久'}",
            )
            self.accept()

    dialog = ActivationDialog()
    result = dialog.exec()
    if result != QDialog.DialogCode.Accepted:
        return False

    return manager.status().valid
