from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QStatusBar,
    QToolBar,
    QFileDialog,
)
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtCore import Qt, QPoint

from dicom.reader import read_dicom
import numpy as np
import os
import pydicom

class MainWindow(QMainWindow):
    """Project Phoenix 医学影像工作站主窗口。"""

    def __init__(self):
        super().__init__()

        # CT 窗宽 / 窗位交互状态
        self.current_dataset = None
        self.current_hu_array = None
        # CT 序列状态
        self.series_files = []
        self.current_slice_index = 0

        self.window_center = None
        self.window_width = None

        self.default_window_center = None
        self.default_window_width = None

        self.windowing_active = False
        self.windowing_start_pos = QPoint()
        self.windowing_start_center = 0.0
        self.windowing_start_width = 1.0

        self.setWindowTitle("Project Phoenix - 医学影像智能工作站")
        self.resize(1400, 900)

        self._build_toolbar()
        self._build_ui()
        self._build_status_bar()

    def _build_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)

        open_action = QAction("打开 DICOM", self)
        open_action.triggered.connect(self._open_dicom)
        toolbar.addAction(open_action)

        open_series_action = QAction("打开 CT 序列", self)
        open_series_action.triggered.connect(self._open_ct_series)
        toolbar.addAction(open_series_action)

        lung_window_action = QAction("肺窗", self)
        lung_window_action.triggered.connect(self._set_lung_window)
        toolbar.addAction(lung_window_action)

        mediastinal_window_action = QAction("纵隔窗", self)
        mediastinal_window_action.triggered.connect(
            self._set_mediastinal_window
        )
        toolbar.addAction(mediastinal_window_action)

        bone_window_action = QAction("骨窗", self)
        bone_window_action.triggered.connect(self._set_bone_window)
        toolbar.addAction(bone_window_action)

        self.addToolBar(toolbar)

    def _build_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        title_label = QLabel("Project Phoenix  医学影像智能工作站")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMinimumHeight(50)

        main_layout.addWidget(title_label)

        workspace_layout = QHBoxLayout()

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setMinimumWidth(220)

        left_layout = QVBoxLayout(left_panel)

        left_title = QLabel("检查 / 序列")
        left_title.setAlignment(Qt.AlignCenter)

        self.left_content_label = QLabel(
            "尚未载入 DICOM\n\n"
            "Study: -\n"
            "Series: -\n"
            "Modality: -"
        )
        self.left_content_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.left_content_label.setWordWrap(True)

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.left_content_label, 1)

        image_panel = QFrame()
        image_panel.setFrameShape(QFrame.StyledPanel)
        image_panel.setMinimumWidth(700)

        image_layout = QVBoxLayout(image_panel)

        image_title = QLabel("影像显示区")
        image_title.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("DICOM Image Viewer\n\n等待载入影像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumSize(400, 400)

        image_layout.addWidget(image_title)
        image_layout.addWidget(self.image_label, 1)

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

    def _display_dicom_image(self, dataset):
        pixel_array = dataset.pixel_array.astype(np.float32)

        if pixel_array.ndim != 2:
            raise ValueError(
                f"当前仅支持二维灰阶影像，实际维度: {pixel_array.shape}"
            )

        modality = getattr(dataset, "Modality", "")

        # 保存当前 DICOM
        self.current_dataset = dataset

        # --------------------------------------------------
        # CT：转换为 HU，并使用 Window Center / Window Width
        # --------------------------------------------------
        if modality == "CT":
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))

            hu_array = pixel_array * slope + intercept
            self.current_hu_array = hu_array

            window_center = getattr(dataset, "WindowCenter", 40.0)
            window_width = getattr(dataset, "WindowWidth", 400.0)

            try:
                window_center = float(window_center[0])
            except (TypeError, IndexError):
                window_center = float(window_center)

            try:
                window_width = float(window_width[0])
            except (TypeError, IndexError):
                window_width = float(window_width)

            if window_width < 1:
                window_width = 1.0

            self.window_center = window_center
            self.window_width = window_width

            self.default_window_center = window_center
            self.default_window_width = window_width

            self._render_ct_window()

        # --------------------------------------------------
        # 非 CT：继续使用 min / max 灰阶显示
        # --------------------------------------------------
        else:
            self.current_hu_array = None

            pixel_min = float(pixel_array.min())
            pixel_max = float(pixel_array.max())

            if pixel_max <= pixel_min:
                raise ValueError("像素值范围无效，无法显示影像")

            image_8bit = (
                (pixel_array - pixel_min)
                / (pixel_max - pixel_min)
                * 255.0
            ).astype(np.uint8)

            photometric = getattr(
                dataset,
                "PhotometricInterpretation",
                "MONOCHROME2",
            )

            if photometric == "MONOCHROME1":
                image_8bit = 255 - image_8bit

            self._show_image_array(image_8bit)


    def _render_ct_window(self):
        """根据当前窗位/窗宽重新渲染 CT。"""
        if self.current_hu_array is None:
            return

        window_width = max(float(self.window_width), 1.0)
        window_center = float(self.window_center)

        window_min = window_center - window_width / 2.0
        window_max = window_center + window_width / 2.0

        image_8bit = np.clip(
            (self.current_hu_array - window_min)
            / (window_max - window_min)
            * 255.0,
            0,
            255,
        ).astype(np.uint8)

        photometric = getattr(
            self.current_dataset,
            "PhotometricInterpretation",
            "MONOCHROME2",
        )

        if photometric == "MONOCHROME1":
            image_8bit = 255 - image_8bit

        self._show_image_array(image_8bit)


    def _show_image_array(self, image_8bit):
        """将 8-bit 灰阶数组显示到中央影像区。"""
        image_8bit = np.ascontiguousarray(image_8bit)

        height, width = image_8bit.shape
        bytes_per_line = image_8bit.strides[0]

        qimage = QImage(
            image_8bit.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Grayscale8,
        ).copy()

        pixmap = QPixmap.fromImage(qimage)
        self.current_pixmap = pixmap

        scaled_pixmap = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setText("")
        self.image_label.setPixmap(scaled_pixmap)

    def _open_ct_series(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择 CT DICOM 序列文件夹",
            ""
        )

        if not folder_path:
            return

        try:
            dicom_files = []

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if not os.path.isfile(file_path):
                    continue

                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                    if getattr(ds, "Modality", "") == "CT":
                        dicom_files.append(file_path)

                except Exception:
                    continue

            if not dicom_files:
                self.statusBar().showMessage(
                    "未找到可读取的 CT DICOM 文件"
                )
                return

            def get_slice_position(file_path):
                try:
                    ds = pydicom.dcmread(
                        file_path,
                        stop_before_pixels=True
                    )

                    if hasattr(ds, "ImagePositionPatient"):
                        return float(ds.ImagePositionPatient[2])

                    if hasattr(ds, "SliceLocation"):
                        return float(ds.SliceLocation)

                    if hasattr(ds, "InstanceNumber"):
                        return float(ds.InstanceNumber)

                except Exception:
                    pass

                return 0.0

            dicom_files.sort(key=get_slice_position)

            self.series_files = dicom_files
            self.current_slice_index = 0

            first_dataset = read_dicom(
                self.series_files[self.current_slice_index]
            )

            self._display_dicom_image(first_dataset)

            modality = getattr(
                first_dataset,
                "Modality",
                "UNKNOWN"
            )
            study = getattr(
                first_dataset,
                "StudyDescription",
                "未提供"
            )
            series = getattr(
                first_dataset,
                "SeriesDescription",
                "未提供"
            )

            self.left_content_label.setText(
                "DICOM 信息\n\n"
                f"Study: {study}\n"
                f"Series: {series}\n"
                f"Modality: {modality}\n"
                f"Images: {len(self.series_files)}\n"
                f"Slice: 1 / {len(self.series_files)}"
            )

            self.statusBar().showMessage(
                f"CT 序列读取成功 | "
                f"{len(self.series_files)} 张 | "
                f"当前 1/{len(self.series_files)}"
            )

        except Exception as exc:
            self.statusBar().showMessage(
                f"CT 序列读取失败：{exc}"
            )
    def _open_dicom(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 DICOM 文件",
            "",
            "DICOM 文件 (*.dcm *.dicom);;所有文件 (*.*)",
        )

        if not file_path:
            return

        try:
            # --------------------------------------------------
            # 进入单张 DICOM 模式：
            # 清除之前加载的 CT 序列，防止滚轮继续翻旧序列
            # --------------------------------------------------
            self.series_files = []
            self.current_slice_index = 0

            dataset = read_dicom(file_path)

            modality = getattr(dataset, "Modality", "UNKNOWN")
            study = getattr(dataset, "StudyDescription", "未提供")
            series = getattr(dataset, "SeriesDescription", "未提供")

            self._display_dicom_image(dataset)

            self.left_content_label.setText(
                "DICOM 信息\n\n"
                f"Study: {study}\n"
                f"Series: {series}\n"
                f"Modality: {modality}"
            )

            self.statusBar().showMessage(
                f"DICOM读取成功 | "
                f"Modality: {modality} | "
                f"Study: {study} | "
                f"Series: {series}"
            )

        except Exception as exc:
            self.statusBar().showMessage(
                f"DICOM读取失败: {exc}"
            )
    def mousePressEvent(self, event):
        if (
            event.button() == Qt.RightButton
            and self.current_hu_array is not None
        ):
            self.windowing_active = True
            self.windowing_start_pos = event.position().toPoint()
            self.windowing_start_center = float(self.window_center)
            self.windowing_start_width = float(self.window_width)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.windowing_active and self.current_hu_array is not None:
            current_pos = event.position().toPoint()

            dx = current_pos.x() - self.windowing_start_pos.x()
            dy = current_pos.y() - self.windowing_start_pos.y()

            self.window_width = max(
                1.0,
                self.windowing_start_width + dx * 2.0,
            )

            self.window_center = (
                self.windowing_start_center - dy * 2.0
            )

            self._render_ct_window()

            self.statusBar().showMessage(
                f"CT 调窗 | WL: {self.window_center:.0f} | "
                f"WW: {self.window_width:.0f}"
            )

            event.accept()
            return

        super().mouseMoveEvent(event)

    def _set_lung_window(self):
        if self.current_hu_array is None:
            self.statusBar().showMessage("请先载入 CT 影像")
            return

        self.window_center = -600.0
        self.window_width = 1500.0

        self._render_ct_window()

        self.statusBar().showMessage(
            f"CT 肺窗 | WL: {self.window_center:.0f} | "
            f"WW: {self.window_width:.0f}"
        )

    def _set_mediastinal_window(self):
        if self.current_hu_array is None:
            self.statusBar().showMessage("请先载入 CT 影像")
            return

        self.window_center = 40.0
        self.window_width = 400.0

        self._render_ct_window()

        self.statusBar().showMessage(
            f"CT 纵隔窗 | WL: {self.window_center:.0f} | "
            f"WW: {self.window_width:.0f}"
        )

    def _set_bone_window(self):
        if self.current_hu_array is None:
            self.statusBar().showMessage("请先载入 CT 影像")
            return

        self.window_center = 300.0
        self.window_width = 1500.0

        self._render_ct_window()

        self.statusBar().showMessage(
            f"CT 骨窗 | WL: {self.window_center:.0f} | "
            f"WW: {self.window_width:.0f}"
        )

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.RightButton
            and self.windowing_active
        ):
            self.windowing_active = False
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.current_hu_array is not None
            and self.default_window_center is not None
            and self.default_window_width is not None
        ):
            self.window_center = float(self.default_window_center)
            self.window_width = float(self.default_window_width)

            self._render_ct_window()

            self.statusBar().showMessage(
                f"CT 默认窗已恢复 | WL: {self.window_center:.0f} | "
                f"WW: {self.window_width:.0f}"
            )

            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if not self.series_files:
            super().wheelEvent(event)
            return

        total_number = len(self.series_files)

        if total_number <= 0:
            event.accept()
            return

        delta = event.angleDelta().y()

        # PACS 风格循环翻层
        # 向上：上一层；第一层继续滚则跳到最后一层
        if delta > 0:
            self.current_slice_index = (
                self.current_slice_index - 1
            ) % total_number

        # 向下：下一层；最后一层继续滚则回到第一层
        elif delta < 0:
            self.current_slice_index = (
                self.current_slice_index + 1
            ) % total_number

        else:
            event.accept()
            return

        # 保留当前窗宽 / 窗位
        previous_window_center = self.window_center
        previous_window_width = self.window_width

        dataset = read_dicom(
            self.series_files[self.current_slice_index]
        )

        self._display_dicom_image(dataset)

        if (
            previous_window_center is not None
            and previous_window_width is not None
            and self.current_hu_array is not None
        ):
            self.window_center = float(previous_window_center)
            self.window_width = float(previous_window_width)
            self._render_ct_window()

        modality = getattr(
            dataset,
            "Modality",
            "UNKNOWN"
        )
        study = getattr(
            dataset,
            "StudyDescription",
            "未提供"
        )
        series = getattr(
            dataset,
            "SeriesDescription",
            "未提供"
        )

        # 内部索引 0～39
        # 界面显示 1～40
        current_number = self.current_slice_index + 1

        self.left_content_label.setText(
            "DICOM 信息\n\n"
            f"Study: {study}\n"
            f"Series: {series}\n"
            f"Modality: {modality}\n"
            f"Images: {total_number}\n"
            f"Slice: {current_number} / {total_number}"
        )

        self.statusBar().showMessage(
            f"CT 序列 | "
            f"当前 {current_number}/{total_number} | "
            f"WL: {self.window_center:.0f} | "
            f"WW: {self.window_width:.0f}"
        )

        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "current_pixmap"):
            scaled_pixmap = self.current_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)