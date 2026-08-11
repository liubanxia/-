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
    QInputDialog,
)
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtCore import Qt, QPoint

from dicom.reader import read_dicom
from dicom.pixel import ct_to_hu, normalize_dx, validate_ct_pixel_dataset
import numpy as np
import os
import pydicom
from pydicom.misc import is_dicom
from pydicom.uid import UID


def _looks_like_ct_dicom_without_preamble(file_path):
    """
    判断无标准 DICM 前导的文件是否具有可信 CT DICOM 结构。

    force=True 仅用于身份鉴别，
    不代表当前 V1 已支持该文件进入正式阅片。
    """

    try:
        dataset = pydicom.dcmread(
            file_path,
            stop_before_pixels=True,
            force=True,
        )
    except Exception:
        return False

    modality = str(
        getattr(dataset, "Modality", "")
    ).strip().upper()

    if modality != "CT":
        return False

    required_uids = (
        "SOPClassUID",
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
    )

    for attribute_name in required_uids:
        value = str(
            getattr(
                dataset,
                attribute_name,
                "",
            )
        ).strip()

        if not value:
            return False

        try:
            uid = UID(value)
        except Exception:
            return False

        if not uid.is_valid:
            return False

    return True


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
        modality = getattr(
            dataset,
            "Modality",
            "",
        )

        # 保存当前 DICOM
        self.current_dataset = dataset

        # --------------------------------------------------
        # CT：统一通过 dicom.pixel 完成安全校验及 HU 转换
        # --------------------------------------------------
        if modality == "CT":
            hu_array = ct_to_hu(dataset)
            self.current_hu_array = hu_array

            window_center = getattr(
                dataset,
                "WindowCenter",
                40.0,
            )

            window_width = getattr(
                dataset,
                "WindowWidth",
                400.0,
            )

            try:
                window_center = float(
                    window_center[0]
                )
            except (TypeError, IndexError):
                window_center = float(
                    window_center
                )

            try:
                window_width = float(
                    window_width[0]
                )
            except (TypeError, IndexError):
                window_width = float(
                    window_width
                )

            if window_width < 1:
                window_width = 1.0

            self.window_center = window_center
            self.window_width = window_width

            self.default_window_center = (
                window_center
            )
            self.default_window_width = (
                window_width
            )

            self._render_ct_window()

        # --------------------------------------------------
        # DX（DR）：统一通过 dicom.pixel 标准化
        # --------------------------------------------------
        elif modality == "DX":
            self.current_hu_array = None

            image_8bit = normalize_dx(
                dataset
            )

            self._show_image_array(
                image_8bit
            )

        else:
            raise ValueError(
                f"当前不支持影像类型：{modality}"
            )


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
            series_groups = {}
            patient_ids = set()
            study_uids = set()

            # ----------------------------------------------
            # 扫描 CT DICOM
            # 先校验病例 / Study 身份，再按 Series 分组
            # ----------------------------------------------
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if not os.path.isfile(file_path):
                    continue

                # ------------------------------------------
                # M8.0-J 文件类型与读取完整性安全门控
                # ------------------------------------------
                try:
                    standard_dicom = is_dicom(
                        file_path
                    )
                except Exception as exc:
                    self.statusBar().showMessage(
                        "安全停止：无法检查文件类型："
                        f"{file_name} | {exc}"
                    )
                    return

                # ------------------------------------------
                # 无 DICM 前导：
                # 普通文件允许忽略；
                # 疑似 CT DICOM 不允许静默跳过。
                # ------------------------------------------
                if not standard_dicom:
                    if _looks_like_ct_dicom_without_preamble(
                        file_path
                    ):
                        self.statusBar().showMessage(
                            "安全停止：检测到无DICM前导的CT DICOM，"
                            "当前V1暂不支持"
                        )
                        return

                    continue

                # ------------------------------------------
                # 已确认标准 DICOM 后，读取失败必须停止。
                # ------------------------------------------
                try:
                    ds = pydicom.dcmread(
                        file_path,
                        stop_before_pixels=True
                    )
                except Exception as exc:
                    self.statusBar().showMessage(
                        "安全停止：标准DICOM读取失败："
                        f"{file_name} | {exc}"
                    )
                    return

                # ------------------------------------------
                # 标准DICOM最小身份结构检查
                # 防止损坏文件被pydicom宽松解析后静默忽略
                # ------------------------------------------
                dataset_sop_class_uid = str(
                    getattr(ds, "SOPClassUID", "")
                ).strip()

                dataset_sop_instance_uid = str(
                    getattr(ds, "SOPInstanceUID", "")
                ).strip()

                file_meta_sop_class_uid = str(
                    getattr(
                        ds.file_meta,
                        "MediaStorageSOPClassUID",
                        "",
                    )
                ).strip()

                file_meta_sop_instance_uid = str(
                    getattr(
                        ds.file_meta,
                        "MediaStorageSOPInstanceUID",
                        "",
                    )
                ).strip()

                effective_sop_class_uid = (
                    dataset_sop_class_uid
                    or file_meta_sop_class_uid
                )

                effective_sop_instance_uid = (
                    dataset_sop_instance_uid
                    or file_meta_sop_instance_uid
                )

                if (
                    not effective_sop_class_uid
                    or not effective_sop_instance_uid
                ):
                    self.statusBar().showMessage(
                        "安全停止：标准DICOM缺失基本SOP身份信息，"
                        "文件可能损坏："
                        f"{file_name}"
                    )
                    return

                # ------------------------------------------
                # CT身份一致性检查
                # 防止Modality缺失时静默漏掉CT切片
                # ------------------------------------------
                modality = str(
                    getattr(ds, "Modality", "")
                ).strip().upper()

                sop_class_uid = str(
                    getattr(ds, "SOPClassUID", "")
                ).strip()

                ct_image_storage_uid = (
                    "1.2.840.10008.5.1.4.1.1.2"
                )

                if modality == "CT":
                    if not sop_class_uid:
                        self.statusBar().showMessage(
                            "安全停止：CT DICOM 缺失 SOPClassUID"
                        )
                        return

                    if sop_class_uid != ct_image_storage_uid:
                        self.statusBar().showMessage(
                            "安全停止：当前V1仅支持标准CT Image Storage"
                        )
                        return

                elif sop_class_uid == ct_image_storage_uid:
                    self.statusBar().showMessage(
                        "安全停止：检测到CT Image Storage，"
                        "但Modality缺失或异常"
                    )
                    return

                else:
                    # 明确不是当前目标CT的其他DICOM允许忽略
                    continue

                patient_id = str(
                    getattr(ds, "PatientID", "")
                ).strip()

                study_uid = str(
                    getattr(ds, "StudyInstanceUID", "")
                ).strip()

                series_uid = str(
                    getattr(ds, "SeriesInstanceUID", "")
                ).strip()

                if not patient_id:
                    self.statusBar().showMessage(
                        "安全停止：CT DICOM 缺失 PatientID"
                    )
                    return

                if not study_uid:
                    self.statusBar().showMessage(
                        "安全停止：CT DICOM 缺失 StudyInstanceUID"
                    )
                    return

                if not series_uid:
                    self.statusBar().showMessage(
                        "安全停止：CT DICOM 缺失 SeriesInstanceUID"
                    )
                    return

                patient_ids.add(patient_id)
                study_uids.add(study_uid)

                if series_uid not in series_groups:
                    series_groups[series_uid] = []

                series_groups[series_uid].append(
                    file_path
                )

            if not series_groups:
                self.statusBar().showMessage(
                    "未找到可读取的 CT DICOM 文件"
                )
                return

            if len(patient_ids) != 1:
                self.statusBar().showMessage(
                    "安全停止：检测到多个 PatientID，禁止混合病例"
                )
                return

            if len(study_uids) != 1:
                self.statusBar().showMessage(
                    "安全停止：检测到多个 StudyInstanceUID，禁止混合检查"
                )
                return

            series_count = len(series_groups)

            # ----------------------------------------------
            # 只有一个 Series：直接进入
            # 多个 Series：必须由医生明确选择
            # ----------------------------------------------
            if series_count == 1:
                selected_series_uid = next(iter(series_groups))
                dicom_files = series_groups[selected_series_uid]

            else:
                series_uids = []
                series_labels = []

                for index, (series_uid, files) in enumerate(
                    series_groups.items(),
                    start=1
                ):
                    description = "未提供"
                    series_number = "未提供"

                    try:
                        header = pydicom.dcmread(
                            files[0],
                            stop_before_pixels=True
                        )

                        description = str(
                            getattr(
                                header,
                                "SeriesDescription",
                                "未提供"
                            )
                        )

                        series_number = str(
                            getattr(
                                header,
                                "SeriesNumber",
                                "未提供"
                            )
                        )

                    except Exception:
                        pass

                    label = (
                        f"{index}. "
                        f"SeriesNumber: {series_number} | "
                        f"{description} | "
                        f"{len(files)} 张"
                    )

                    series_uids.append(series_uid)
                    series_labels.append(label)

                selected_label, ok = QInputDialog.getItem(
                    self,
                    "选择 CT Series",
                    "检测到多个 CT Series，请由医生明确选择：",
                    series_labels,
                    0,
                    False
                )

                if not ok:
                    self.statusBar().showMessage(
                        "已取消 CT Series 选择，未进入阅片"
                    )
                    return

                selected_index = series_labels.index(
                    selected_label
                )

                selected_series_uid = series_uids[
                    selected_index
                ]

                dicom_files = series_groups[
                    selected_series_uid
                ]

           # ----------------------------------------------
            # M8.0-H CT切片唯一性与空间排序安全门控
            # ----------------------------------------------
            import math

            slice_records = []
            sop_uids = set()

            reference_row = None
            reference_col = None
            reference_normal = None

            for file_path in dicom_files:
                try:
                    ds = pydicom.dcmread(
                        file_path,
                        stop_before_pixels=True
                    )
                except Exception:
                    self.statusBar().showMessage(
                        "安全停止：CT切片头信息读取失败"
                    )
                    return

                # ------------------------------------------
                # SOPInstanceUID 唯一性检查
                # ------------------------------------------
                sop_uid = str(
                    getattr(ds, "SOPInstanceUID", "")
                ).strip()

                if not sop_uid:
                    self.statusBar().showMessage(
                        "安全停止：CT切片缺失 SOPInstanceUID"
                    )
                    return

                if sop_uid in sop_uids:
                    self.statusBar().showMessage(
                        "安全停止：检测到重复 SOPInstanceUID"
                    )
                    return

                sop_uids.add(sop_uid)

                # ------------------------------------------
                # 必须存在真实空间定位信息
                # ------------------------------------------
                if not hasattr(ds, "ImagePositionPatient"):
                    self.statusBar().showMessage(
                        "安全停止：CT切片缺失 ImagePositionPatient"
                    )
                    return

                if not hasattr(ds, "ImageOrientationPatient"):
                    self.statusBar().showMessage(
                        "安全停止：CT切片缺失 ImageOrientationPatient"
                    )
                    return

                try:
                    ipp = [
                        float(v)
                        for v in ds.ImagePositionPatient
                    ]

                    iop = [
                        float(v)
                        for v in ds.ImageOrientationPatient
                    ]

                    if len(ipp) != 3 or len(iop) != 6:
                        raise ValueError

                    if not all(
                        math.isfinite(v)
                        for v in ipp + iop
                    ):
                        raise ValueError

                except Exception:
                    self.statusBar().showMessage(
                        "安全停止：CT切片空间定位信息异常"
                    )
                    return

                # ------------------------------------------
                # 计算并标准化行、列方向向量
                # ------------------------------------------
                row = iop[:3]
                col = iop[3:]

                row_length = math.sqrt(
                    sum(v * v for v in row)
                )

                col_length = math.sqrt(
                    sum(v * v for v in col)
                )

                if row_length < 1e-6 or col_length < 1e-6:
                    self.statusBar().showMessage(
                        "安全停止：CT图像方向向量异常"
                    )
                    return

                row = [
                    v / row_length
                    for v in row
                ]

                col = [
                    v / col_length
                    for v in col
                ]

                # 行、列方向应近似正交
                row_col_dot = sum(
                    row[i] * col[i]
                    for i in range(3)
                )

                if abs(row_col_dot) > 1e-3:
                    self.statusBar().showMessage(
                        "安全停止：CT图像方向向量不正交"
                    )
                    return

                # ------------------------------------------
                # 第一张切片建立参考方向
                # ------------------------------------------
                if reference_row is None:
                    reference_row = row
                    reference_col = col

                    normal = [
                        row[1] * col[2]
                        - row[2] * col[1],

                        row[2] * col[0]
                        - row[0] * col[2],

                        row[0] * col[1]
                        - row[1] * col[0],
                    ]

                    normal_length = math.sqrt(
                        sum(v * v for v in normal)
                    )

                    if normal_length < 1e-6:
                        self.statusBar().showMessage(
                            "安全停止：无法确定CT切片法向量"
                        )
                        return

                    reference_normal = [
                        v / normal_length
                        for v in normal
                    ]

                else:
                    # --------------------------------------
                    # 同一Series方向必须保持一致
                    # --------------------------------------
                    row_difference = max(
                        abs(
                            row[i]
                            - reference_row[i]
                        )
                        for i in range(3)
                    )

                    col_difference = max(
                        abs(
                            col[i]
                            - reference_col[i]
                        )
                        for i in range(3)
                    )

                    if (
                        row_difference > 1e-3
                        or col_difference > 1e-3
                    ):
                        self.statusBar().showMessage(
                            "安全停止：CT Series内图像方向不一致"
                        )
                        return

                # ------------------------------------------
                # IPP 投影到切片法向量
                # 得到真正的空间排序位置
                # ------------------------------------------
                slice_position = sum(
                    ipp[i] * reference_normal[i]
                    for i in range(3)
                )

                slice_records.append(
                    (
                        slice_position,
                        file_path,
                    )
                )

            # ----------------------------------------------
            # 按真实空间位置排序
            # ----------------------------------------------
            slice_records.sort(
                key=lambda item: item[0]
            )

            # ----------------------------------------------
            # 检查重复空间位置
            # ----------------------------------------------
            for index in range(
                1,
                len(slice_records)
            ):
                previous_position = (
                    slice_records[index - 1][0]
                )

                current_position = (
                    slice_records[index][0]
                )

                if abs(
                    current_position
                    - previous_position
                ) < 1e-3:
                    self.statusBar().showMessage(
                        "安全停止：检测到重复CT切片空间位置"
                    )
                    return

            dicom_files = [
                item[1]
                for item in slice_records
            ]

            # ----------------------------------------------
            # M8.0-I CT全Series像素与HU标定安全预检查
            # ----------------------------------------------
            for file_path in dicom_files:
                try:
                    dataset = read_dicom(
                        file_path
                    )

                    validate_ct_pixel_dataset(
                        dataset
                    )

                except Exception as exc:
                    self.statusBar().showMessage(
                        "安全停止：CT像素或HU标定检查失败："
                        f"{exc}"
                    )
                    return

            self.series_files = dicom_files
            self.current_slice_index = 0

            first_dataset = read_dicom(
                self.series_files[
                    self.current_slice_index
                ]
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

            series_number = getattr(
                first_dataset,
                "SeriesNumber",
                "未提供"
            )

            self.left_content_label.setText(
                "DICOM 信息\n\n"
                f"Study: {study}\n"
                f"Series: {series}\n"
                f"SeriesNumber: {series_number}\n"
                f"Modality: {modality}\n"
                f"Images: {len(self.series_files)}\n"
                f"Slice: 1 / {len(self.series_files)}\n"
                f"CT Series: {series_count}"
            )

            self.statusBar().showMessage(
                f"CT 序列读取成功 | "
                f"{len(self.series_files)} 张 | "
                f"检测到 {series_count} 个 CT Series | "
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

        # 保存翻层前索引，运行时异常时恢复
        previous_slice_index = self.current_slice_index

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

        # 保留翻层前显示与CT状态
        previous_window_center = self.window_center
        previous_window_width = self.window_width
        previous_dataset = self.current_dataset
        previous_hu_array = self.current_hu_array

        previous_default_window_center = getattr(
            self,
            "default_window_center",
            None,
        )

        previous_default_window_width = getattr(
            self,
            "default_window_width",
            None,
        )

        try:
            dataset = read_dicom(
                self.series_files[
                    self.current_slice_index
                ]
            )

            self._display_dicom_image(
                dataset
            )

            if (
                previous_window_center is not None
                and previous_window_width is not None
                and self.current_hu_array is not None
            ):
                self.window_center = float(
                    previous_window_center
                )
                self.window_width = float(
                    previous_window_width
                )
                self._render_ct_window()

        except Exception as exc:
            # 恢复到翻层前状态，禁止索引与显示影像错位
            self.current_slice_index = (
                previous_slice_index
            )
            self.current_dataset = (
                previous_dataset
            )
            self.current_hu_array = (
                previous_hu_array
            )
            self.window_center = (
                previous_window_center
            )
            self.window_width = (
                previous_window_width
            )
            self.default_window_center = (
                previous_default_window_center
            )
            self.default_window_width = (
                previous_default_window_width
            )

            self.statusBar().showMessage(
                "安全停止：CT切片读取或显示失败："
                f"{exc}"
            )

            event.accept()
            return

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