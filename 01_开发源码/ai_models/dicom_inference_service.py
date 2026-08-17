
from pathlib import Path
import json
import numpy as np
import pydicom

try:
    from .model_manager import PhoenixModelManager
    from .ct_bodypart_adapter import BodyPartRegressionCTAdapter
except ImportError:
    from model_manager import PhoenixModelManager
    from ct_bodypart_adapter import BodyPartRegressionCTAdapter


class PhoenixDICOMInferenceService:

    XRAY_MODALITIES = {
        "DX", "DR", "CR", "XR"
    }

    def __init__(self, base_dir=None):

        if base_dir is None:
            base_dir = Path(__file__).resolve().parent

        self.base_dir = Path(base_dir)

        self.registry_path = (
            self.base_dir / "模型注册表.json"
        )

        self.config_path = (
            self.base_dir / "MVP调用链.json"
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.config = json.load(f)

        self.manager = PhoenixModelManager(
            self.registry_path
        )

        self.active = False
        self.current_file = None
        self.current_modality = None
        self.metadata = None

        # 这里只建立适配器对象；
        # BodyPartRegression模型本身仍然延迟加载
        self.ct_adapter = BodyPartRegressionCTAdapter(
            project_root=self.base_dir.parent.parent,
            repo_path=Path(r"D:\project_phoenix\04_AI模型\工程工作区\模型拆解\BodyPartRegression\BodyPartRegression-develop"),
            model_dir=Path(r"D:\project_phoenix\04_AI模型\路由模型\BodyPartRegression\weights\public_bpr_model\public_bpr_model"),
        )

    # ========================================================
    # DICOM只读元数据
    # 不读取姓名、PatientID等患者标识信息
    # ========================================================
    def inspect_dicom(self, dicom_path):

        path = Path(dicom_path)

        probe_path = None
        ds = None

        if path.is_dir():

            for candidate in sorted(path.rglob("*")):

                if not candidate.is_file():
                    continue

                try:
                    candidate_ds = pydicom.dcmread(
                        str(candidate),
                        stop_before_pixels=True,
                        force=True
                    )
                except Exception:
                    continue

                modality = str(
                    getattr(
                        candidate_ds,
                        "Modality",
                        ""
                    )
                ).upper().strip()

                if modality in {
                    "CT",
                    "DX",
                    "CR",
                    "DR",
                    "MG",
                }:
                    probe_path = candidate
                    ds = candidate_ds
                    break

            if ds is None:
                raise RuntimeError(
                    f"目录中没有可识别的CT/DR DICOM: {path}"
                )

        else:

            probe_path = path

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True
            )

        modality = str(
            getattr(ds, "Modality", "")
        ).upper().strip()

        data = {
            "modality": modality,
            "body_part": str(
                getattr(ds, "BodyPartExamined", "")
            ),
            "study_description": str(
                getattr(ds, "StudyDescription", "")
            ),
            "series_description": str(
                getattr(ds, "SeriesDescription", "")
            ),
            "protocol_name": str(
                getattr(ds, "ProtocolName", "")
            ),
            "rows": int(
                getattr(ds, "Rows", 0) or 0
            ),
            "columns": int(
                getattr(ds, "Columns", 0) or 0
            ),
        }

        return data

    # ========================================================
    # 医生主动启动
    # ========================================================
    def open_case(
        self,
        dicom_path,
        doctor_confirmed=False
    ):

        if doctor_confirmed is not True:
            raise PermissionError(
                "医生未确认启动AI，禁止进入推理流程。"
            )

        metadata = self.inspect_dicom(
            dicom_path
        )

        modality = metadata["modality"]

        if modality == "CT":
            route = "CT"

        elif modality in self.XRAY_MODALITIES:
            route = "DR"

        else:
            raise ValueError(
                f"Phoenix MVP暂不支持模态："
                f"{modality or 'UNKNOWN'}"
            )

        self.manager.activate_for_case()

        self.active = True
        original_path = Path(dicom_path)

        if route == "CT":
            # CT keeps the complete YUNPACS case directory.
            self.current_file = original_path

        else:
            # DR inference requires one actual DICOM file.
            if original_path.is_dir():

                selected = None

                for candidate in sorted(
                    original_path.rglob("*")
                ):
                    if not candidate.is_file():
                        continue

                    try:
                        test_ds = pydicom.dcmread(
                            str(candidate),
                            stop_before_pixels=True,
                            force=True
                        )
                    except Exception:
                        continue

                    test_modality = str(
                        getattr(
                            test_ds,
                            "Modality",
                            ""
                        )
                    ).upper().strip()

                    if test_modality in self.XRAY_MODALITIES:
                        selected = candidate
                        break

                if selected is None:
                    raise RuntimeError(
                        "YUNPACS病例目录中没有可用DR DICOM"
                    )

                self.current_file = selected

            else:
                self.current_file = original_path

        self.current_modality = route
        self.metadata = metadata

        return {
            "status": "AI_SESSION_ACTIVE",
            "route": route,
            "metadata": metadata,
            "plan": self.config[route],
        }

    # ========================================================
    # DR像素标准化
    # ========================================================
    def _load_xray_image(self):

        ds = pydicom.dcmread(
            str(self.current_file),
            force=True
        )

        arr = ds.pixel_array.astype(
            np.float32
        )

        # 避免极端值影响显示/模型输入
        low = np.percentile(arr, 1)
        high = np.percentile(arr, 99)

        if high <= low:
            low = float(arr.min())
            high = float(arr.max())

        if high > low:
            arr = np.clip(
                arr,
                low,
                high
            )

            arr = (
                (arr - low)
                / (high - low)
                * 255.0
            )
        else:
            arr = np.zeros_like(arr)

        arr = arr.astype(np.uint8)

        photometric = str(
            getattr(
                ds,
                "PhotometricInterpretation",
                ""
            )
        ).upper()

        if photometric == "MONOCHROME1":
            arr = 255 - arr

        # 灰度 → 三通道
        if arr.ndim == 2:
            arr = np.stack(
                [arr, arr, arr],
                axis=-1
            )

        return arr

    # ========================================================
    # YOLO结果转换为Phoenix统一结构
    # ========================================================
    def _parse_yolo_result(
        self,
        model_name,
        result
    ):

        findings = []

        boxes = getattr(
            result,
            "boxes",
            None
        )

        names = getattr(
            result,
            "names",
            {}
        )

        if boxes is not None:

            for box in boxes:

                cls_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                xyxy = [
                    round(float(x), 2)
                    for x in
                    box.xyxy[0].tolist()
                ]

                if isinstance(names, dict):
                    label = str(
                        names.get(
                            cls_id,
                            cls_id
                        )
                    )
                else:
                    label = str(cls_id)

                findings.append({
                    "type": "detection",
                    "model": model_name,
                    "class_id": cls_id,
                    "label": label,
                    "confidence": round(
                        confidence,
                        5
                    ),
                    "bbox_xyxy": xyxy,
                })

        masks = getattr(
            result,
            "masks",
            None
        )

        # ----------------------------------------------------
        # Instance segmentation真实轮廓
        #
        # Ultralytics Masks.xy：
        # 每个实例对应一个原始图像像素坐标polygon。
        # 转成纯Python float，便于Phoenix UI和JSON使用。
        # ----------------------------------------------------
        mask_payloads = []

        if masks is not None:

            try:
                polygons = masks.xy
            except Exception:
                polygons = []

            for mask_index, polygon in enumerate(
                polygons
            ):

                try:
                    points = np.asarray(
                        polygon,
                        dtype=float
                    )
                except Exception:
                    continue

                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or points.shape[0] < 3
                    or not np.isfinite(points).all()
                ):
                    continue

                # 非常复杂的轮廓做轻量降采样，
                # 避免Qt绘制和结果JSON过大。
                max_points = 512

                if len(points) > max_points:
                    step = int(
                        np.ceil(
                            len(points)
                            / max_points
                        )
                    )

                    points = points[::step]

                label = ""
                confidence = 0.0
                class_id = None

                # segmentation实例与boxes顺序一致时，
                # 同步类别及confidence。
                try:
                    if (
                        boxes is not None
                        and mask_index < len(boxes)
                    ):
                        box = boxes[mask_index]

                        class_id = int(
                            box.cls[0].item()
                        )

                        confidence = float(
                            box.conf[0].item()
                        )

                        if isinstance(names, dict):
                            label = str(
                                names.get(
                                    class_id,
                                    class_id
                                )
                            )
                        else:
                            label = str(
                                class_id
                            )

                except Exception:
                    pass

                mask_payloads.append({
                    "type": "segmentation",
                    "model": model_name,
                    "class_id": class_id,
                    "label": label,
                    "confidence": round(
                        confidence,
                        5
                    ),
                    "polygon_xy": [
                        [
                            round(float(x), 2),
                            round(float(y), 2)
                        ]
                        for x, y in points
                    ],
                })

        mask_count = len(
            mask_payloads
        )

        return {
            "model": model_name,
            "finding_count": len(findings),
            "mask_count": int(mask_count),
            "findings": findings,
            "masks": mask_payloads,
        }

    # ========================================================
    # DR视觉B真实推理
    # ========================================================
    def _run_dr_visual_b(self):

        image = self._load_xray_image()

        models = self.config[
            "DR"
        ].get(
            "视觉B",
            []
        )

        outputs = []

        for model_name in models:

            model = self.manager.load_model(
                model_name
            )

            result = model.predict(
                source=image,
                verbose=False
            )[0]

            outputs.append(
                self._parse_yolo_result(
                    model_name,
                    result
                )
            )

        return outputs

    # ========================================================
    # 当前病例运行
    # ========================================================
    def run_current_case(self):

        if not self.active:
            raise RuntimeError(
                "医生尚未启动AI，禁止推理。"
            )

        base_result = {
            "phoenix_schema": "1.0",
            "modality_route": (
                self.current_modality
            ),
            "metadata": self.metadata,
            "patient_identifiers_exported": False,
            "ai_outputs": [],
        }

        # ----------------------------------------------------
        # DR
        # ----------------------------------------------------
        if self.current_modality == "DR":

            base_result[
                "ai_outputs"
            ] = self._run_dr_visual_b()

            base_result[
                "status"
            ] = "DR_INFERENCE_COMPLETE"

            return base_result

        # ----------------------------------------------------
        # CT
        #
        # BodyPartRegression已部署，但其原生slice
        # 预处理/输出映射必须按模型自身接口适配。
        # 此处不猜、不伪造。
        # ----------------------------------------------------
        if self.current_modality == "CT":

            ct_output = self.ct_adapter.run(
                self.current_file
            )

            base_result["ai_outputs"] = [
                ct_output
            ]

            base_result[
                "status"
            ] = "CT_ROUTING_COMPLETE"

            return base_result

    # ========================================================
    # 医生结束病例
    # ========================================================
    def close_case(self):

        self.ct_adapter.release()
        self.manager.deactivate_case()

        self.active = False
        self.current_file = None
        self.current_modality = None
        self.metadata = None

        return {
            "status": "AI_SESSION_CLOSED"
        }
