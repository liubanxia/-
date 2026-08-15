from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QListWidget,
    QPushButton,
    QStatusBar,
    QToolBar,
    QFileDialog,
    QInputDialog,
)
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QPen
from PySide6.QtCore import Qt, QPoint

from dicom.reader import read_dicom
from dicom.pixel import ct_to_hu, normalize_dx, validate_ct_pixel_dataset
from ai.dual_vision_controller import DualVisionController
from ai.fracture_candidate_store import FractureCandidateStore
from ai.fracture_candidate import FractureCandidate
from ai.dual_vision_orchestrator import DualVisionOrchestrator
from ai.mock_visuals import MockVisualA, MockVisualB
import numpy as np
import os
import pydicom
from pydicom.misc import is_dicom
from pydicom.uid import UID
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QTextEdit, QTabWidget


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

        # 双视觉 AI 默认关闭，只有医生主动操作后才允许启动。
        self.dual_vision_controller = DualVisionController()

        # 视觉B骨折候选容器。
        # 候选只属于当前影像上下文。
        self.fracture_candidate_store = FractureCandidateStore()

        # 当前正在由医生复核的视觉B候选。
        # None表示当前没有激活的候选叠加。
        self.active_fracture_candidate = None
        self.active_fracture_candidate_index = None

        # 视觉B候选医生复核状态。
        # key为当前FractureCandidateStore中的候选索引。
        # value仅允许：
        # pending / accepted / rejected
        self.fracture_candidate_review_status = {}

        # M9.0-A：
        # 当前使用 Mock 视觉模型，只验证双通路调度，
        # 不执行真实医学影像诊断。
        self.visual_a = MockVisualA()
        self.visual_b = MockVisualB()

        self.dual_vision_orchestrator = DualVisionOrchestrator(
            controller=self.dual_vision_controller,
            visual_a=self.visual_a,
            visual_b=self.visual_b,
        )

        # CT 窗宽 / 窗位交互状态
        self.current_dataset = None
        self.current_hu_array = None

        # 当前用于显示及视觉模型输入的8-bit二维影像。
        # CT为当前窗宽窗位渲染结果，DX为normalize_dx标准化结果。
        self.current_image_array = None
        # CT 序列状态
        self.series_files = []
        self.current_dicom_path = None
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

        # 双视觉 AI 默认不可启动。
        # 只有 CT Series 完成全部安全门控后才允许医生点击。
        self.dual_vision_action = QAction("启动双视觉AI", self)
        self.dual_vision_action.setEnabled(False)
        self.dual_vision_action.triggered.connect(
            self._toggle_dual_vision_ai
        )
        toolbar.addSeparator()
        toolbar.addAction(self.dual_vision_action)
        # DR segmentation Mask显示开关。
        self._phoenix_mask_overlay_visible = True

        self.phoenix_mask_overlay_action = QAction(
            "显示Mask",
            self
        )

        self.phoenix_mask_overlay_action.setCheckable(
            True
        )

        self.phoenix_mask_overlay_action.setChecked(
            True
        )

        self.phoenix_mask_overlay_action.triggered.connect(
            self._toggle_phoenix_mask_overlay
        )

        toolbar.addAction(
            self.phoenix_mask_overlay_action
        )

        # Phoenix右侧AI辅助阅片结果区
        self._setup_phoenix_result_panel()

        self.addToolBar(toolbar)

    def _setup_phoenix_result_panel(self):
        """
        Phoenix右侧AI辅助阅片面板。

        这里只展示AI辅助信息。
        不自动写入最终诊断，不替代医生审核。
        """
        self.phoenix_result_dock = QDockWidget(
            "Phoenix AI辅助阅片",
            self
        )

        self.phoenix_result_dock.setObjectName(
            "PhoenixAIResultDock"
        )

        container = QWidget(
            self.phoenix_result_dock
        )

        layout = QVBoxLayout(container)

        self.phoenix_result_header = QLabel(
            "AI待机 | 打开病例后由医生点击“启动双视觉AI”"
        )

        self.phoenix_result_header.setWordWrap(
            True
        )

        layout.addWidget(
            self.phoenix_result_header
        )

        self.phoenix_result_tabs = QTabWidget(
            container
        )

        # ----------------------------------------------------
        # 1. AI分析结果
        # ----------------------------------------------------
        self.phoenix_analysis_text = QTextEdit()
        self.phoenix_analysis_text.setReadOnly(
            True
        )

        self.phoenix_analysis_text.setPlainText(
            "尚未运行AI。"
        )

        self.phoenix_result_tabs.addTab(
            self.phoenix_analysis_text,
            "AI分析结果"
        )

        # ----------------------------------------------------
        # 2. 结构化报告草稿
        # ----------------------------------------------------
        self.phoenix_report_text = QTextEdit()

        self.phoenix_report_text.setPlaceholderText(
            "AI完成后生成结构化辅助草稿。\n"
            "最终报告必须由医生审核、修改和确认。"
        )

        self.phoenix_result_tabs.addTab(
            self.phoenix_report_text,
            "结构化报告"
        )

        # ----------------------------------------------------
        # 3. 医生审核
        # ----------------------------------------------------
        self.phoenix_review_text = QTextEdit()

        self.phoenix_review_text.setPlaceholderText(
            "医生审核区：\n"
            "可记录AI错误、遗漏、修改原因等。\n"
            "本阶段暂不自动写入学习库。"
        )

        self.phoenix_result_tabs.addTab(
            self.phoenix_review_text,
            "医生审核"
        )

        layout.addWidget(
            self.phoenix_result_tabs
        )

        container.setLayout(layout)

        self.phoenix_result_dock.setWidget(
            container
        )

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.phoenix_result_dock
        )

        # 默认显示，但内容为空。
        self.phoenix_result_dock.show()


    def _sync_phoenix_dr_segmentation_masks(
        self,
        result
    ):
        """
        将Phoenix DR segmentation polygon保存为
        当前病例的独立显示覆盖层。

        不写回DICOM。
        """
        if not isinstance(result, dict):
            self._phoenix_segmentation_masks = []
            return 0

        if str(
            result.get(
                "modality_route",
                ""
            )
        ).upper() != "DR":
            self._phoenix_segmentation_masks = []
            return 0
        dataset = getattr(
            self,
            "current_dataset",
            None
        )

        if dataset is None:
            self._phoenix_segmentation_masks = []
            return 0

        sop_uid = str(
            getattr(
                dataset,
                "SOPInstanceUID",
                ""
            )
        ).strip()

        payloads = []

        outputs = result.get(
            "ai_outputs",
            []
        ) or []

        for output in outputs:

            model_name = str(
                output.get(
                    "model",
                    ""
                )
            )

            masks = output.get(
                "masks",
                []
            ) or []

            for mask in masks:

                polygon = mask.get(
                    "polygon_xy"
                )

                try:
                    coordinates = np.asarray(
                        polygon,
                        dtype=float
                    )
                except Exception:
                    continue

                if (
                    coordinates.ndim != 2
                    or coordinates.shape[1] != 2
                    or len(coordinates) < 3
                    or not np.isfinite(
                        coordinates
                    ).all()
                ):
                    continue

                payloads.append({
                    "sop_instance_uid":
                        sop_uid,
                    "model":
                        model_name,
                    "label":
                        str(
                            mask.get(
                                "label",
                                ""
                            )
                        ),
                    "confidence":
                        float(
                            mask.get(
                                "confidence",
                                0.0
                            )
                            or 0.0
                        ),
                    "polygon_xy":
                        coordinates.tolist(),
                })

        self._phoenix_segmentation_masks = (
            payloads
        )

        self._render_current_pixmap()

        return len(payloads)


    def _sync_phoenix_dr_fracture_candidates(
        self,
        result
    ):
        """
        将Phoenix DR视觉B bbox结果转换成现有
        FractureCandidate并注入候选系统。

        只修改显示候选状态，不修改原始DICOM。
        """

        if not isinstance(result, dict):
            return 0

        route = str(
            result.get(
                "modality_route",
                ""
            )
        ).upper()

        if route != "DR":
            return 0

        dataset = getattr(
            self,
            "current_dataset",
            None
        )

        if dataset is None:
            return 0

        sop_uid = str(
            getattr(
                dataset,
                "SOPInstanceUID",
                ""
            )
        ).strip()

        if not sop_uid:
            return 0

        # DR是单幅影像，当前slice索引固定使用现有上下文索引。
        slice_index = getattr(
            self,
            "current_slice_index",
            0
        )

        if (
            not isinstance(slice_index, int)
            or isinstance(slice_index, bool)
            or slice_index < 0
        ):
            slice_index = 0

        # 获取当前原始显示尺寸，用于bbox安全裁剪。
        image_width = None
        image_height = None

        pixmap = getattr(
            self,
            "current_pixmap",
            None
        )

        if (
            pixmap is not None
            and not pixmap.isNull()
        ):
            image_width = int(
                pixmap.width()
            )
            image_height = int(
                pixmap.height()
            )

        image_array = getattr(
            self,
            "current_image_array",
            None
        )

        if (
            image_array is not None
            and getattr(
                image_array,
                "ndim",
                0
            ) >= 2
        ):
            image_height = int(
                image_array.shape[0]
            )
            image_width = int(
                image_array.shape[1]
            )

        candidates_with_meta = []

        outputs = result.get(
            "ai_outputs",
            []
        ) or []

        for output in outputs:

            model_name = str(
                output.get(
                    "model",
                    "视觉B"
                )
            )

            findings = output.get(
                "findings",
                []
            ) or []

            for finding in findings:

                bbox = finding.get(
                    "bbox_xyxy"
                )

                if bbox is None:
                    continue

                try:
                    coordinates = np.asarray(
                        bbox,
                        dtype=float
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    continue

                if coordinates.shape != (4,):
                    continue

                if not np.isfinite(
                    coordinates
                ).all():
                    continue

                x1, y1, x2, y2 = [
                    float(x)
                    for x in coordinates
                ]
                # 在写入候选之前先限制到真实影像范围。
                if (
                    image_width is not None
                    and image_height is not None
                    and image_width > 0
                    and image_height > 0
                ):
                    x1 = max(
                        0.0,
                        min(
                            x1,
                            image_width - 1
                        )
                    )

                    y1 = max(
                        0.0,
                        min(
                            y1,
                            image_height - 1
                        )
                    )

                    x2 = max(
                        0.0,
                        min(
                            x2,
                            image_width - 1
                        )
                    )

                    y2 = max(
                        0.0,
                        min(
                            y2,
                            image_height - 1
                        )
                    )

                if (
                    x2 <= x1
                    or y2 <= y1
                ):
                    continue

                try:
                    confidence = float(
                        finding.get(
                            "confidence",
                            0.0
                        )
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    continue

                if not np.isfinite(
                    confidence
                ):
                    continue

                confidence = max(
                    0.0,
                    min(
                        confidence,
                        1.0
                    )
                )

                try:
                    candidate = FractureCandidate(
                        slice_index=slice_index,
                        sop_instance_uid=sop_uid,
                        confidence=confidence,
                        region_type="bbox",
                        region=(
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                    )
                except Exception:
                    continue

                candidates_with_meta.append(
                    (
                        candidate,
                        {
                            "model":
                                model_name,
                            "label":
                                str(
                                    finding.get(
                                        "label",
                                        ""
                                    )
                                ),
                            "confidence":
                                confidence,
                        }
                    )
                )

        # 高置信度候选优先显示。
        candidates_with_meta.sort(
            key=lambda x:
                x[0].confidence,
            reverse=True
        )

        candidates = [
            x[0]
            for x in candidates_with_meta
        ]

        metadata = [
            x[1]
            for x in candidates_with_meta
        ]

        # 原子替换上一病例/上一轮AI候选。
        count = (
            self.fracture_candidate_store.replace_all(
                candidates
            )
        )

        self._phoenix_dr_candidate_metadata = (
            metadata
        )

        self.active_fracture_candidate = None
        self.active_fracture_candidate_index = None

        self.fracture_candidate_review_status.clear()

        self.accept_fracture_candidate_button.setEnabled(
            False
        )

        self.reject_fracture_candidate_button.setEnabled(
            False
        )

        self._refresh_fracture_candidate_list()

        # 若存在候选，默认激活最高置信度候选。
        if count > 0:

            current_candidates = (
                self.fracture_candidate_store.get_all()
            )

            first = current_candidates[0]

            self.active_fracture_candidate = first
            self.active_fracture_candidate_index = 0

            self.accept_fracture_candidate_button.setEnabled(
                True
            )

            self.reject_fracture_candidate_button.setEnabled(
                True
            )

            try:
                self.fracture_candidate_list.setCurrentRow(
                    0
                )
            except Exception:
                pass

        # 统一通过现有渲染链重绘。
        self._render_current_pixmap()

        return count


    def _render_phoenix_ai_result(
        self,
        result
    ):
        """
        把Phoenix统一结果显示到右侧面板。

        不读取患者姓名/PatientID。
        """
        if not isinstance(result, dict):
            self.phoenix_result_header.setText(
                "AI结果格式异常"
            )

            self.phoenix_analysis_text.setPlainText(
                str(result)
            )

            return

        route = str(
            result.get(
                "modality_route",
                ""
            )
        ).upper()

        outputs = result.get(
            "ai_outputs",
            []
        ) or []

        analysis_lines = []
        report_lines = []

        # ====================================================
        # CT
        # ====================================================
        if route == "CT":

            self.phoenix_result_header.setText(
                "Phoenix CT AI分析完成"
            )

            analysis_lines.append(
                "【CT身体部位路由】"
            )

            if outputs:

                item = outputs[0]

                display = (
                    item.get(
                        "body_part_display"
                    )
                    or item.get(
                        "body_part_examined_tag"
                    )
                    or "部位未确定"
                )

                raw_tag = item.get(
                    "body_part_examined_tag_raw"
                )

                regions = item.get(
                    "active_body_regions",
                    []
                )

                analysis_lines.extend([
                    f"模型：{item.get('model', 'BodyPartRegression')}",
                    f"Phoenix判定部位：{display}",
                    f"模型原始Tag：{raw_tag}",
                    f"活动区域：{regions}",
                    f"CT切片数：{item.get('slice_count', '')}",
                    f"矩阵：{item.get('matrix', '')}",
                    f"像素间距(mm)：{item.get('pixel_spacings_mm', '')}",
                    f"Valid Z-spacing：{item.get('valid_z_spacing', '')}",
                    "",
                    "说明：BodyPartRegression当前用于CT身体部位/层面路由，"
                    "不等同于病灶诊断。"
                ])

                report_lines.extend([
                    "【检查类型】",
                    "CT",
                    "",
                    "【AI部位路由】",
                    str(display),
                    "",
                    "【AI影像学所见】",
                    "当前阶段仅完成身体部位路由，"
                    "尚未由通用视觉A生成病灶级影像学所见。",
                    "",
                    "【诊断意见】",
                    "待医生阅片及后续视觉模型结果。",
                ])

            else:
                analysis_lines.append(
                    "BodyPartRegression未返回有效输出。"
                )

        # ====================================================
        # DR / X-ray
        # ====================================================
        elif route == "DR":

            self.phoenix_result_header.setText(
                "Phoenix DR AI分析完成"
            )

            # 将bbox结果接入既有视觉B候选列表和覆盖层。
            dr_candidate_count = (
                self._sync_phoenix_dr_fracture_candidates(
                    result
                )
            )

            dr_mask_overlay_count = (
                self._sync_phoenix_dr_segmentation_masks(
                    result
                )
            )

            analysis_lines.append(
                "【DR视觉B：骨折漏诊防护】"
            )

            total_findings = 0
            total_masks = 0

            for item in outputs:

                model_name = item.get(
                    "model",
                    "Unknown"
                )

                finding_count = int(
                    item.get(
                        "finding_count",
                        0
                    )
                    or 0
                )

                mask_count = int(
                    item.get(
                        "mask_count",
                        0
                    )
                    or 0
                )

                total_findings += finding_count
                total_masks += mask_count

                analysis_lines.extend([
                    "",
                    f"模型：{model_name}",
                    f"检测候选：{finding_count}",
                    f"Mask：{mask_count}",
                ])

                findings = item.get(
                    "findings",
                    []
                ) or []

                for index, finding in enumerate(
                    findings[:20],
                    start=1
                ):
                    analysis_lines.append(
                        "  "
                        f"{index}. "
                        f"{finding.get('label', '')} | "
                        f"confidence="
                        f"{finding.get('confidence', '')} | "
                        f"bbox="
                        f"{finding.get('bbox_xyxy', '')}"
                    )

            analysis_lines.extend([
                "",
                "--------------------------------",
                f"总检测候选：{total_findings}",
                f"已注入视觉B候选列表：{dr_candidate_count}",
                f"已注入Mask覆盖层：{dr_mask_overlay_count}",
                f"总Mask：{total_masks}",
                "",
                "说明：以上为视觉B安全防护候选，"
                "不得直接等同于最终骨折诊断。"
            ])

            report_lines.extend([
                "【检查类型】",
                "DR / X-ray",
                "",
                "【AI骨折安全防护】",
                f"候选检测数：{total_findings}",
                f"分割Mask数：{total_masks}",
                "",
                "【AI影像学所见】",
            ])

            if total_findings:
                report_lines.append(
                    "视觉B提示存在骨折候选区域，"
                    "请医生结合原始影像逐一复核。"
                )
            else:
                report_lines.append(
                    "视觉B当前未检出超过默认阈值的骨折候选。"
                )

            report_lines.extend([
                "",
                "【诊断意见】",
                "由医生审核原始影像后填写。",
            ])

        # ====================================================
        # 其他
        # ====================================================
        else:

            self.phoenix_result_header.setText(
                "Phoenix AI分析完成"
            )

            analysis_lines.append(
                f"未知模态路由：{route}"
            )

            analysis_lines.append(
                str(result)
            )

        self.phoenix_analysis_text.setPlainText(
            "\n".join(
                str(x)
                for x in analysis_lines
            )
        )

        # 只覆盖AI草稿区；
        # 医生审核区永远不自动覆盖。
        self.phoenix_report_text.setPlainText(
            "\n".join(
                str(x)
                for x in report_lines
            )
        )

        self.phoenix_result_tabs.setCurrentWidget(
            self.phoenix_analysis_text
        )

        self.phoenix_result_dock.show()
        self.phoenix_result_dock.raise_()


    def _render_phoenix_ai_error(
        self,
        error
    ):
        """
        AI运行错误显示。
        """
        if not isinstance(error, dict):
            error = {
                "message": str(error)
            }

        message = str(
            error.get(
                "message",
                "未知错误"
            )
        )

        self.phoenix_result_header.setText(
            "Phoenix AI运行失败"
        )

        self.phoenix_analysis_text.setPlainText(
            "AI运行失败\n\n"
            f"{message}"
        )

        self.phoenix_result_dock.show()

    def _ensure_phoenix_ai_controller(self):
        """
        延迟创建Phoenix AI控制器。
        创建控制器本身不会加载模型。
        """
        controller = getattr(
            self,
            "_phoenix_ai_controller",
            None
        )

        if controller is not None:
            return controller

        from pathlib import Path
        import sys

        source_root = (
            Path(__file__).resolve().parents[1]
        )

        if str(source_root) not in sys.path:
            sys.path.insert(
                0,
                str(source_root)
            )

        from ui_agent.phoenix_ai_button_controller import (
            PhoenixAIButtonController
        )

        controller = PhoenixAIButtonController(
            source_root / "ai_models"
        )

        self._phoenix_ai_controller = controller

        timer = QTimer(self)
        timer.setInterval(250)
        timer.timeout.connect(
            self._poll_phoenix_ai
        )

        self._phoenix_ai_poll_timer = timer

        return controller


    def _current_phoenix_dicom_path(self):
        """
        返回当前病例实际DICOM路径。
        """
        path = getattr(
            self,
            "current_dicom_path",
            None
        )

        if path:
            return path

        series_files = getattr(
            self,
            "series_files",
            []
        )

        if series_files:
            index = getattr(
                self,
                "current_slice_index",
                0
            )

            if not isinstance(index, int):
                index = 0

            index = max(
                0,
                min(
                    index,
                    len(series_files) - 1
                )
            )

            return series_files[index]

        dataset = getattr(
            self,
            "current_dataset",
            None
        )

        if dataset is not None:
            filename = getattr(
                dataset,
                "filename",
                None
            )

            if filename:
                return filename

        return None


    def _toggle_dual_vision_ai(self):
        """
        医生主动启动Phoenix AI。

        打开病例不会触发本函数。
        """
        try:
            controller = (
                self._ensure_phoenix_ai_controller()
            )

            if controller.is_running():
                self.statusBar().showMessage(
                    "AI正在分析，请等待当前任务结束"
                )
                return

            dicom_path = (
                self._current_phoenix_dicom_path()
            )

            if not dicom_path:
                self.statusBar().showMessage(
                    "当前没有可供AI处理的DICOM病例"
                )
                return

            self._phoenix_ai_case_token = str(
                dicom_path
            )

            # 只登记病例
            controller.set_current_case(
                dicom_path
            )

            # 此处才是真正的医生主动触发点
            controller.click_ai_button()

            self.dual_vision_action.setText(
                "AI分析中…"
            )

            self.dual_vision_action.setEnabled(
                False
            )

            self.statusBar().showMessage(
                "医生已启动Phoenix AI，正在分析…"
            )

            self.phoenix_result_header.setText(
                "Phoenix AI正在分析…"
            )

            self.phoenix_analysis_text.setPlainText(
                "模型运行中，请等待分析完成。"
            )

            self._phoenix_ai_poll_timer.start()

        except Exception as exc:

            self.dual_vision_action.setText(
                "启动双视觉AI"
            )

            self.dual_vision_action.setEnabled(
                True
            )

            self.statusBar().showMessage(
                f"Phoenix AI启动失败：{exc}"
            )


    def _poll_phoenix_ai(self):
        """
        Qt主线程轮询AI工作线程。
        工作线程不直接修改Qt控件。
        """
        controller = getattr(
            self,
            "_phoenix_ai_controller",
            None
        )

        if controller is None:
            return

        if controller.is_running():
            return

        timer = getattr(
            self,
            "_phoenix_ai_poll_timer",
            None
        )

        if timer is not None:
            timer.stop()

        launched_case = getattr(
            self,
            "_phoenix_ai_case_token",
            None
        )

        current_case = (
            self._current_phoenix_dicom_path()
        )

        same_case = (
            launched_case is not None
            and current_case is not None
            and str(current_case)
            == launched_case
        )

        if (
            controller.state
            == controller.STATE_COMPLETE
        ):
            result = controller.get_result()

            self.phoenix_ai_result = result

            # 将AI结果送入右侧辅助阅片面板
            self._render_phoenix_ai_result(
                result
            )

            if same_case:
                self.statusBar().showMessage(
                    self._summarize_phoenix_ai_result(
                        result
                    )
                )
            else:
                self.statusBar().showMessage(
                    "AI分析完成，但当前病例已经切换；"
                    "上一病例结果未显示"
                )

        elif (
            controller.state
            == controller.STATE_ERROR
        ):
            error = (
                controller.get_error()
                or {}
            )

            self.statusBar().showMessage(
                "Phoenix AI分析失败："
                + str(
                    error.get(
                        "message",
                        "未知错误"
                    )
                )
            )

            self._render_phoenix_ai_error(
                error
            )

        self.dual_vision_action.setText(
            "启动双视觉AI"
        )

        self.dual_vision_action.setEnabled(
            self._current_phoenix_dicom_path()
            is not None
        )


    def _summarize_phoenix_ai_result(
        self,
        result
    ):
        """
        将Phoenix结果转换为主界面状态栏摘要。
        """
        if not isinstance(result, dict):
            return "Phoenix AI分析完成"

        route = str(
            result.get(
                "modality_route",
                ""
            )
        ).upper()

        outputs = result.get(
            "ai_outputs",
            []
        )

        if route == "CT":

            if outputs:
                item = outputs[0]

                body = (
                    item.get(
                        "body_part_display"
                    )
                    or item.get(
                        "body_part_examined_tag"
                    )
                    or "部位未确定"
                )

                return (
                    "Phoenix CT AI完成 | "
                    f"BodyPart: {body}"
                )

            return "Phoenix CT AI完成"

        if route == "DR":

            finding_count = 0
            mask_count = 0

            for item in outputs:

                finding_count += int(
                    item.get(
                        "finding_count",
                        0
                    )
                    or 0
                )

                mask_count += int(
                    item.get(
                        "mask_count",
                        0
                    )
                    or 0
                )

            return (
                "Phoenix DR AI完成 | "
                f"模型: {len(outputs)} | "
                f"检测: {finding_count} | "
                f"Mask: {mask_count}"
            )

        return "Phoenix AI分析完成"

    def _apply_visual_b_inference_result(
        self,
        vision_b_result,
    ):
        """
        安全接收一轮视觉B推理结果。

        只有视觉B明确成功并返回标准
        FractureCandidate tuple时，
        才允许替换当前候选。

        推理失败、解析失败或格式异常时，
        必须保留医生当前已经看到的旧候选。
        """

        if not isinstance(
            vision_b_result,
            dict,
        ):
            self.statusBar().showMessage(
                "视觉B结果未采用：通路结果格式异常，"
                "保留上一轮候选"
            )
            return False

        error = vision_b_result.get(
            "error"
        )

        if error is not None:
            self.statusBar().showMessage(
                "视觉B本轮推理失败，"
                "保留上一轮有效候选"
            )
            return False

        candidates = vision_b_result.get(
            "result"
        )

        # VisualBOutputParser标准输出固定为tuple。
        # 防止原始ONNX输出、Mock字典或其他未解析结果
        # 被错误注入候选Store。
        if not isinstance(
            candidates,
            tuple,
        ):
            self.statusBar().showMessage(
                "视觉B结果未采用："
                "尚未转换为标准候选批次，"
                "保留上一轮候选"
            )
            return False

        if not all(
            isinstance(
                candidate,
                FractureCandidate,
            )
            for candidate in candidates
        ):
            self.statusBar().showMessage(
                "视觉B结果未采用："
                "候选批次包含非法对象，"
                "保留上一轮候选"
            )
            return False

        try:
            count = self._replace_visual_b_candidates(
                candidates
            )
        except Exception as exc:
            self.statusBar().showMessage(
                "视觉B候选更新失败，"
                "保留上一轮有效候选："
                f"{exc}"
            )
            return False

        self.statusBar().showMessage(
            "视觉B本轮候选已安全更新 | "
            f"候选数量：{count}"
        )

        return True

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

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setMinimumWidth(300)

        right_layout = QVBoxLayout(right_panel)

        fracture_title = QLabel("视觉B骨折候选")
        fracture_title.setAlignment(Qt.AlignCenter)

        self.fracture_candidate_list = QListWidget()
        self.fracture_candidate_list.setMinimumHeight(220)
        self.fracture_candidate_list.itemClicked.connect(
            self._on_fracture_candidate_clicked
        )

        fracture_review_layout = QHBoxLayout()

        self.accept_fracture_candidate_button = QPushButton(
            "确认候选"
        )
        self.accept_fracture_candidate_button.setEnabled(
            False
        )
        self.accept_fracture_candidate_button.clicked.connect(
            self._accept_active_fracture_candidate
        )

        self.reject_fracture_candidate_button = QPushButton(
            "否定候选"
        )
        self.reject_fracture_candidate_button.setEnabled(
            False
        )
        self.reject_fracture_candidate_button.clicked.connect(
            self._reject_active_fracture_candidate
        )

        fracture_review_layout.addWidget(
            self.accept_fracture_candidate_button
        )
        fracture_review_layout.addWidget(
            self.reject_fracture_candidate_button
        )

        report_title = QLabel("AI / 报告")
        report_title.setAlignment(Qt.AlignCenter)

        report_placeholder = QLabel(
            "AI分析结果\n\n"
            "结构化报告\n\n"
            "医生审核区"
        )
        report_placeholder.setAlignment(Qt.AlignCenter)
        report_placeholder.setWordWrap(True)

        right_layout.addWidget(fracture_title)
        right_layout.addWidget(
            self.fracture_candidate_list,
            1,
        )
        right_layout.addLayout(
            fracture_review_layout
        )
        right_layout.addWidget(report_title)
        right_layout.addWidget(
            report_placeholder,
            1,
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


    def _draw_phoenix_segmentation_overlay(
        self,
        display_pixmap
    ):
        """
        在pixmap副本上绘制半透明segmentation polygon。

        原始DICOM、current_pixmap和HU数组均不修改。
        """
        if display_pixmap is None:
            return display_pixmap

        if not getattr(
            self,
            "_phoenix_mask_overlay_visible",
            True
        ):
            return display_pixmap

        masks = getattr(
            self,
            "_phoenix_segmentation_masks",
            []
        ) or []

        if not masks:
            return display_pixmap

        dataset = getattr(
            self,
            "current_dataset",
            None
        )

        if dataset is None:
            return display_pixmap

        current_sop = str(
            getattr(
                dataset,
                "SOPInstanceUID",
                ""
            )
        ).strip()

        from PySide6.QtCore import QPointF
        from PySide6.QtGui import (
            QColor,
            QBrush,
            QPolygonF,
        )

        painter = QPainter(
            display_pixmap
        )
        try:
            # 半透明填充 + 边缘
            fill_color = QColor(
                255, 0, 255, 70
            )

            edge_color = QColor(
                255, 0, 255, 210
            )

            pen = QPen(
                edge_color
            )

            pen.setWidth(2)

            painter.setPen(
                pen
            )

            painter.setBrush(
                QBrush(
                    fill_color
                )
            )

            width = (
                display_pixmap.width()
            )

            height = (
                display_pixmap.height()
            )

            for mask in masks:

                if (
                    str(
                        mask.get(
                            "sop_instance_uid",
                            ""
                        )
                    ).strip()
                    != current_sop
                ):
                    continue

                points = mask.get(
                    "polygon_xy",
                    []
                )

                polygon = QPolygonF()

                for point in points:

                    try:
                        x = float(
                            point[0]
                        )

                        y = float(
                            point[1]
                        )
                    except Exception:
                        continue

                    if not (
                        np.isfinite(x)
                        and np.isfinite(y)
                    ):
                        continue

                    x = max(
                        0.0,
                        min(
                            x,
                            width - 1
                        )
                    )

                    y = max(
                        0.0,
                        min(
                            y,
                            height - 1
                        )
                    )

                    polygon.append(
                        QPointF(
                            x,
                            y
                        )
                    )

                if polygon.count() >= 3:
                    painter.drawPolygon(
                        polygon
                    )

        finally:
            painter.end()

        return display_pixmap


    def _toggle_phoenix_mask_overlay(
        self,
        checked
    ):
        """
        医生显示/隐藏Mask覆盖层。
        """
        self._phoenix_mask_overlay_visible = bool(
            checked
        )

        self._render_current_pixmap()


    def _build_fracture_overlay_pixmap(self):
        """
        基于原始current_pixmap生成视觉B候选显示副本。

        不修改原始DICOM、HU数组或current_pixmap。
        当前只支持bbox：
        (x1, y1, x2, y2)，坐标基于原始影像像素。
        """

        if (
            not hasattr(self, "current_pixmap")
            or self.current_pixmap.isNull()
        ):
            return None

        # 永远在副本上绘制，不修改原始影像pixmap。
        display_pixmap = self.current_pixmap.copy()

        # 先绘制segmentation半透明层，
        # 再由现有逻辑绘制活动bbox。
        display_pixmap = (
            self._draw_phoenix_segmentation_overlay(
                display_pixmap
            )
        )

        candidate = self.active_fracture_candidate

        if candidate is None:
            return display_pixmap

        if not isinstance(candidate, FractureCandidate):
            return display_pixmap

        # 当前阶段只支持bbox。
        if candidate.region_type != "bbox":
            return display_pixmap

        # 候选必须属于当前CT层。
        if candidate.slice_index != self.current_slice_index:
            return display_pixmap

        dataset = self.current_dataset

        if dataset is None:
            return display_pixmap

        actual_sop_uid = str(
            getattr(
                dataset,
                "SOPInstanceUID",
                "",
            )
        ).strip()

        # 显示层再次进行SOP身份核对。
        if (
            actual_sop_uid
            != candidate.sop_instance_uid
        ):
            return display_pixmap

        try:
            coordinates = np.asarray(
                candidate.region,
                dtype=float,
            )
        except (TypeError, ValueError):
            return display_pixmap

        if coordinates.shape != (4,):
            return display_pixmap

        if not np.isfinite(coordinates).all():
            return display_pixmap

        x1, y1, x2, y2 = coordinates

        image_width = display_pixmap.width()
        image_height = display_pixmap.height()

        if image_width <= 0 or image_height <= 0:
            return display_pixmap

        # bbox必须具有正面积。
        if x2 <= x1 or y2 <= y1:
            return display_pixmap

        # 限制到当前真实影像边界。
        x1 = max(
            0.0,
            min(x1, image_width - 1),
        )
        y1 = max(
            0.0,
            min(y1, image_height - 1),
        )
        x2 = max(
            0.0,
            min(x2, image_width - 1),
        )
        y2 = max(
            0.0,
            min(y2, image_height - 1),
        )

        if x2 <= x1 or y2 <= y1:
            return display_pixmap

        painter = QPainter(
            display_pixmap
        )

        try:
            pen = QPen(Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)

            painter.drawRect(
                int(round(x1)),
                int(round(y1)),
                max(
                    1,
                    int(round(x2 - x1)),
                ),
                max(
                    1,
                    int(round(y2 - y1)),
                ),
            )
        finally:
            painter.end()

        return display_pixmap

    def _render_current_pixmap(self):
        """
        统一显示原始影像及当前可用视觉叠加。
        """
        display_pixmap = (
            self._build_fracture_overlay_pixmap()
        )

        if display_pixmap is None:
            return

        scaled_pixmap = display_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setText("")
        self.image_label.setPixmap(
            scaled_pixmap
        )

    def _show_image_array(self, image_8bit):
        """将 8-bit 灰阶数组显示到中央影像区。"""
        image_8bit = np.ascontiguousarray(image_8bit)

        # 保存当前实际显示的二维影像快照，
        # 供视觉模型显式启动后构建输入。
        self.current_image_array = image_8bit.copy()

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

        self._render_current_pixmap()

    def _reset_dual_vision_for_context_change(self):
        """
        病例、Study 或 Series 即将变化时调用。

        新影像上下文不得继承上一病例的 AI 激活状态。
        """

        self.dual_vision_controller.reset_for_context_change()

        # 病例/Study/Series变化后立即使旧AI影像输入失效。
        self.current_image_array = None

        # 新病例 / Study / Series不得继承旧骨折候选。
        self.fracture_candidate_store.clear()
        self.fracture_candidate_list.clear()
        self.active_fracture_candidate = None
        self.active_fracture_candidate_index = None
        self.fracture_candidate_review_status.clear()

        # 新病例不得继承上一病例Mask。
        self._phoenix_segmentation_masks = []

        # 上下文失效后立即移除旧候选叠加，
        # 即使新影像随后加载失败，也不得残留旧bbox。
        if hasattr(self, "current_pixmap"):
            self._render_current_pixmap()

        self.accept_fracture_candidate_button.setEnabled(
            False
        )
        self.reject_fracture_candidate_button.setEnabled(
            False
        )

        self.dual_vision_action.setText("启动双视觉AI")
        self.dual_vision_action.setEnabled(False)

    def _on_fracture_candidate_clicked(self, item):
        """
        医生点击右侧视觉B候选时执行。

        列表项只负责选择候选；
        真正跳层仍由FractureCandidate身份门控负责。
        """

        row = self.fracture_candidate_list.row(
            item
        )

        candidates = (
            self.fracture_candidate_store.get_all()
        )

        if not 0 <= row < len(candidates):
            self.statusBar().showMessage(
                "安全停止：骨折候选列表索引异常"
            )
            return

        candidate = candidates[row]

        self.accept_fracture_candidate_button.setEnabled(
            False
        )
        self.reject_fracture_candidate_button.setEnabled(
            False
        )

        # 点击新的候选前先取消旧候选叠加，
        # 防止定位失败时继续显示旧框造成误导。
        self.active_fracture_candidate = None
        self.active_fracture_candidate_index = None
        self._render_current_pixmap()

        success = self._jump_to_fracture_candidate(
            candidate
        )

        if not success:
            return

        # 只有真实CT定位及SOP身份核对成功后，
        # 才允许将该候选设为当前医生复核对象。
        self.active_fracture_candidate = candidate
        self.active_fracture_candidate_index = row
        self._render_current_pixmap()

        self.accept_fracture_candidate_button.setEnabled(
            True
        )
        self.reject_fracture_candidate_button.setEnabled(
            True
        )

    def _review_active_fracture_candidate(
        self,
        status,
    ):
        """
        保存医生对当前视觉B候选的复核意见。

        只修改医生复核状态，
        不修改模型原始FractureCandidate。
        """

        candidate = self.active_fracture_candidate
        candidate_index = (
            self.active_fracture_candidate_index
        )

        candidates = (
            self.fracture_candidate_store.get_all()
        )

        valid_index = (
            isinstance(candidate_index, int)
            and not isinstance(candidate_index, bool)
            and 0 <= candidate_index < len(candidates)
        )

        if (
            candidate is None
            or not valid_index
            or candidates[candidate_index] is not candidate
        ):
            self.active_fracture_candidate = None
            self.active_fracture_candidate_index = None

            self.accept_fracture_candidate_button.setEnabled(
                False
            )
            self.reject_fracture_candidate_button.setEnabled(
                False
            )

            self._render_current_pixmap()

            self.statusBar().showMessage(
                "安全停止：当前视觉B复核候选身份失效"
            )
            return False

        try:
            self._set_fracture_candidate_review_status(
                candidate_index,
                status,
            )
        except Exception as exc:
            self.statusBar().showMessage(
                "安全停止：视觉B候选复核状态写入失败："
                f"{exc}"
            )
            return False

        self._refresh_fracture_candidate_list()

        # 刷新列表后恢复当前医生复核项的选中位置。
        self.fracture_candidate_list.setCurrentRow(
            candidate_index
        )

        if status == "accepted":
            status_text = "医生已确认保留候选"
        elif status == "rejected":
            status_text = "医生已否定候选"
        else:
            status_text = "候选恢复待复核"

        self.statusBar().showMessage(
            f"视觉B候选复核：{status_text} | "
            f"Slice: {candidate.slice_index + 1} | "
            f"Confidence: {candidate.confidence:.3f}"
        )

        return True

    def _accept_active_fracture_candidate(self):
        """
        医生确认保留当前视觉B候选。
        """

        return self._review_active_fracture_candidate(
            "accepted"
        )

    def _reject_active_fracture_candidate(self):
        """
        医生否定当前视觉B候选。
        """
        return self._review_active_fracture_candidate(
            "rejected"
        )

    def _get_fracture_candidate_review_status(
        self,
        candidate_index,
    ):
        """
        返回指定视觉B候选的医生复核状态。
        """

        if (
            not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
        ):
            raise TypeError(
                "候选索引必须是整数"
            )

        if candidate_index < 0:
            raise ValueError(
                "候选索引不能小于0"
            )

        candidates = (
            self.fracture_candidate_store.get_all()
        )

        if candidate_index >= len(candidates):
            raise IndexError(
                "候选索引超出当前视觉B候选范围"
            )

        return self.fracture_candidate_review_status.get(
            candidate_index,
            "pending",
        )

    def _set_fracture_candidate_review_status(
        self,
        candidate_index,
        status,
    ):
        """
        设置医生对视觉B候选的复核状态。

        仅记录医生复核结果，
        不修改模型原始FractureCandidate。
        """

        if (
            not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
        ):
            raise TypeError(
                "候选索引必须是整数"
            )

        if candidate_index < 0:
            raise ValueError(
                "候选索引不能小于0"
            )

        candidates = (
            self.fracture_candidate_store.get_all()
        )

        if candidate_index >= len(candidates):
            raise IndexError(
                "候选索引超出当前视觉B候选范围"
            )

        if status not in (
            "pending",
            "accepted",
            "rejected",
        ):
            raise ValueError(
                "候选复核状态仅允许"
                "pending、accepted或rejected"
            )

        if status == "pending":
            self.fracture_candidate_review_status.pop(
                candidate_index,
                None,
            )
        else:
            self.fracture_candidate_review_status[
                candidate_index
            ] = status

        return status

    def _replace_visual_b_candidates(
        self,
        candidates,
    ):
        """
        将一轮新的视觉B候选原子注入当前CT上下文。

        安全顺序：
        1. 先由Store完整验证并替换；
        2. 只有替换成功后才清理上一轮UI复核状态；
        3. 最后刷新右侧候选列表。

        新批次非法时，旧候选及旧UI状态保持不变。
        """

        count = (
            self.fracture_candidate_store.replace_all(
                candidates
            )
        )

        # Store替换成功后，上一轮候选上下文失效。
        self.active_fracture_candidate = None
        self.active_fracture_candidate_index = None
        self.fracture_candidate_review_status.clear()

        self.accept_fracture_candidate_button.setEnabled(
            False
        )
        self.reject_fracture_candidate_button.setEnabled(
            False
        )

        # 立即移除上一轮候选bbox。
        if hasattr(self, "current_pixmap"):
            self._render_current_pixmap()

        self._refresh_fracture_candidate_list()

        self.statusBar().showMessage(
            "视觉B候选已更新 | "
            f"候选数量：{count}"
        )

        return count

    def _refresh_fracture_candidate_list(self):
        """
        将当前视觉B骨折候选同步到右侧列表。

        当前只负责显示候选，
        不执行跳层、不修改候选结果。
        """

        self.fracture_candidate_list.clear()

        candidates = (
            self.fracture_candidate_store.get_all()
        )

        for index, candidate in enumerate(
            candidates
        ):
            review_status = (
                self._get_fracture_candidate_review_status(
                    index
                )
            )

            if review_status == "accepted":
                review_text = "已保留"
            elif review_status == "rejected":
                review_text = "已否定"
            else:
                review_text = "待复核"

            display_text = (
                f"{index + 1}. "
                f"Slice {candidate.slice_index + 1} | "
                f"Confidence {candidate.confidence:.3f} | "
                f"{review_text}"
            )

            self.fracture_candidate_list.addItem(
                display_text
            )

    def _resolve_fracture_candidate_dataset(self, candidate):
        """
        将视觉B骨折候选严格绑定到当前CT Series中的真实切片。

        安全规则：
        1. 必须是FractureCandidate；
        2. 当前必须存在CT Series；
        3. slice_index必须位于当前Series范围内；
        4. 对应切片的SOPInstanceUID必须与候选完全一致；
        5. 只有全部一致时才返回真实DICOM dataset。
        """

        if not isinstance(candidate, FractureCandidate):
            raise TypeError(
                "候选跳层仅允许FractureCandidate"
            )

        if not self.series_files:
            raise RuntimeError(
                "当前没有可用于候选定位的CT Series"
            )

        slice_index = candidate.slice_index

        if slice_index >= len(self.series_files):
            raise IndexError(
                "骨折候选slice_index超出当前CT Series范围"
            )

        file_path = self.series_files[
            slice_index
        ]

        dataset = read_dicom(
            file_path
        )

        actual_sop_uid = str(
            getattr(
                dataset,
                "SOPInstanceUID",
                "",
            )
        ).strip()

        if not actual_sop_uid:
            raise RuntimeError(
                "候选对应CT切片缺失SOPInstanceUID"
            )

        if actual_sop_uid != candidate.sop_instance_uid:
            raise RuntimeError(
                "骨折候选与当前CT切片SOPInstanceUID不一致，禁止跳转"
            )

        return dataset

    def _jump_to_fracture_candidate(self, candidate):
        """
        跳转到视觉B骨折候选对应的真实CT切片。

        流程：
        1. 严格核对candidate类型；
        2. 核对slice_index范围；
        3. 核对SOPInstanceUID；
        4. 身份一致后调用统一CT切片显示入口；
        5. 不自行创建第二套切片显示逻辑。
        """

        try:
            # 先完成候选与真实DICOM身份核对。
            self._resolve_fracture_candidate_dataset(
                candidate
            )

            # 身份确认后，才允许显示对应层。
            dataset = self._show_ct_series_slice(
                candidate.slice_index
            )

        except Exception as exc:
            self.statusBar().showMessage(
                "安全停止：骨折候选定位失败："
                f"{exc}"
            )
            return False

        self.statusBar().showMessage(
            "视觉B骨折候选定位成功 | "
            f"Slice: {candidate.slice_index + 1}"
            f"/{len(self.series_files)} | "
            f"Confidence: {candidate.confidence:.3f}"
        )

        return True

    def _open_ct_series(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择 CT DICOM 序列文件夹",
            ""
        )

        if not folder_path:
            return

        # 新 CT 上下文开始：
        # 立即停止旧病例 AI，并保持按钮禁用。
        self._reset_dual_vision_for_context_change()

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

            # Phoenix正式AI入口：
            # 加载病例时只登记DICOM，绝不启动AI。
            self.current_dicom_path = (
                self.series_files[0]
                if self.series_files
                else None
            )

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

            # 到这里说明：
            # 病例 / Study / Series / 空间排序 /
            # 像素 / HU / 首层显示均已成功。
            # 此时才允许医生主动点击启动双视觉 AI。
            self.dual_vision_action.setEnabled(True)

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

        # 真正进入新的单张 DICOM 上下文后，
        # 立即清除上一 CT Series 的双视觉 AI 状态。
        self._reset_dual_vision_for_context_change()

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

            # Phoenix DR正式AI入口：
            # 打开影像只登记病例，不启动模型。
            self.current_dataset = dataset
            self.current_dicom_path = file_path

            if str(modality).strip().upper() in {
                "DX", "DR", "CR", "XR"
            }:
                self.dual_vision_action.setText(
                    "启动双视觉AI"
                )
                self.dual_vision_action.setEnabled(
                    True
                )

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

    def _show_ct_series_slice(self, target_index):
        """
        显示当前CT Series中的指定切片。

        统一负责：
        - 索引范围检查；
        - DICOM读取；
        - 保持当前窗宽窗位；
        - 显示失败时恢复原状态；
        - 更新左侧Slice信息；
        - 更新状态栏。

        后续滚轮翻层与骨折候选跳层统一调用本方法。
        """

        if not self.series_files:
            raise RuntimeError(
                "当前没有已加载的CT Series"
            )

        if not isinstance(target_index, int):
            raise TypeError(
                "CT切片索引必须是整数"
            )

        total_number = len(self.series_files)

        if not 0 <= target_index < total_number:
            raise IndexError(
                "CT切片索引超出当前Series范围"
            )

        previous_slice_index = self.current_slice_index
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
                    target_index
                ]
            )

            self.current_slice_index = (
                target_index
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

        except Exception:
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
            raise

        modality = getattr(
            dataset,
            "Modality",
            "UNKNOWN",
        )

        study = getattr(
            dataset,
            "StudyDescription",
            "未提供",
        )

        series = getattr(
            dataset,
            "SeriesDescription",
            "未提供",
        )

        current_number = (
            self.current_slice_index + 1
        )

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

        return dataset

    def wheelEvent(self, event):
        """
        CT Series滚轮翻层。

        实际切片读取、显示、窗宽窗位保持和失败回滚，
        统一交给_show_ct_series_slice()处理。
        """

        if not self.series_files:
            super().wheelEvent(event)
            return

        total_number = len(self.series_files)

        if total_number <= 0:
            event.accept()
            return

        delta = event.angleDelta().y()

        if delta > 0:
            target_index = (
                self.current_slice_index - 1
            ) % total_number

        elif delta < 0:
            target_index = (
                self.current_slice_index + 1
            ) % total_number

        else:
            event.accept()
            return

        try:
            self._show_ct_series_slice(
                target_index
            )

        except Exception as exc:
            self.statusBar().showMessage(
                "安全停止：CT切片读取或显示失败："
                f"{exc}"
            )

        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "current_pixmap"):
            self._render_current_pixmap()
