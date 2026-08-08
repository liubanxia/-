from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QStatusBar,
)
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    """Project Phoenix 医学影像工作站主窗口。"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Project Phoenix - 医学影像智能工作站")
        self.resize(1400, 900)

        self._build_ui()
        self._build_status_bar()

    def _build_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # 顶部标题区
        title_label = QLabel("Project Phoenix  医学影像智能工作站")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMinimumHeight(50)

        main_layout.addWidget(title_label)

        # 主工作区
        workspace_layout = QHBoxLayout()

        # 左侧：检查 / 序列区
        left_panel = self._create_panel(
            "检查 / 序列",
            "患者与检查列表\n\n后续显示：\nStudy\nSeries\nModality",
            220,
        )

        # 中央：影像显示区
        image_panel = self._create_panel(
            "影像显示区",
            "DICOM Image Viewer\n\n等待载入影像",
            700,
        )

        # 右侧：AI / 报告区
        right_panel = self._create_panel(
            "AI / 报告",
            "AI分析结果\n\n结构化报告\n\n医生审核区",
            300,
        )

        workspace_layout.addWidget(left_panel)
        workspace_layout.addWidget(image_panel, 1)
        workspace_layout.addWidget(right_panel)

        main_layout.addLayout(workspace_layout, 1)

        self.setCentralWidget(central_widget)

    def _create_panel(
        self,
        title: str,
        content: str,
        minimum_width: int,
    ) -> QFrame:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMinimumWidth(minimum_width)

        layout = QVBoxLayout(panel)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)

        content_label = QLabel(content)
        content_label.setAlignment(Qt.AlignCenter)
        content_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(content_label, 1)

        return panel

    def _build_status_bar(self):
        status_bar = QStatusBar()
        status_bar.showMessage("Project Phoenix 已启动 | 等待载入影像")
        self.setStatusBar(status_bar)