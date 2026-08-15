
from pathlib import Path
import sys
import os
import gc
import numpy as np
import pydicom


class BodyPartRegressionCTAdapter:
    """
    Project Phoenix CT BodyPartRegression适配器

    原则：
    - 使用MIC-DKFZ原生InferenceModel.npy2json
    - DICOM像素先转换为HU
    - DICOM序列按空间位置排序
    - 不读取/输出患者姓名和PatientID
    - 构造xyz体数据后交给模型官方预处理
    """

    def __init__(
        self,
        project_root,
        repo_path,
        model_dir
    ):
        self.project_root = Path(project_root)
        self.repo_path = Path(repo_path)
        self.model_dir = Path(model_dir)

        self.model = None

    # --------------------------------------------------------
    # 延迟加载：只有医生已启动且真正运行CT时才加载
    # --------------------------------------------------------
    def _ensure_model(self):

        if self.model is not None:
            return self.model

        repo = str(self.repo_path)

        if repo not in sys.path:
            sys.path.insert(0, repo)

        from bpreg.inference.inference_model import (
            InferenceModel
        )

        base_dir = str(self.model_dir)

        if not base_dir.endswith(
            ("/", "\\")
        ):
            base_dir += os.sep

        # Phoenix当前医院部署优先CPU
        self.model = InferenceModel(
            base_dir=base_dir,
            gpu=False,
            warning_to_error=True
        )

        return self.model

    # --------------------------------------------------------
    # DICOM单张排序位置
    # 对斜轴位也使用ImageOrientationPatient计算法向
    # --------------------------------------------------------
    @staticmethod
    def _spatial_position(ds, normal=None):

        try:
            ipp = np.asarray(
                ds.ImagePositionPatient,
                dtype=float
            )

            if normal is not None:
                return float(
                    np.dot(ipp, normal)
                )

            return float(ipp[2])

        except Exception:
            pass

        try:
            return float(ds.SliceLocation)
        except Exception:
            pass

        try:
            return float(ds.InstanceNumber)
        except Exception:
            return 0.0

    # --------------------------------------------------------
    # 找到一个完整CT Series
    # 输入可以是一张DICOM，也可以是目录
    # --------------------------------------------------------
    def _find_series(self, input_path):

        input_path = Path(input_path)

        target_uid = None

        if input_path.is_file():

            first = pydicom.dcmread(
                str(input_path),
                stop_before_pixels=True,
                force=True
            )

            if str(
                getattr(first, "Modality", "")
            ).upper() != "CT":
                raise ValueError(
                    "输入文件不是CT DICOM"
                )

            target_uid = str(
                getattr(
                    first,
                    "SeriesInstanceUID",
                    ""
                )
            )

            scan_root = input_path.parent

        else:
            scan_root = input_path

        groups = {}

        for p in scan_root.rglob("*"):

            if not p.is_file():
                continue

            try:
                ds = pydicom.dcmread(
                    str(p),
                    stop_before_pixels=True,
                    force=True
                )

                if str(
                    getattr(
                        ds,
                        "Modality",
                        ""
                    )
                ).upper() != "CT":
                    continue

                # 排除常见定位像
                image_type = "\\".join(
                    str(x).upper()
                    for x in getattr(
                        ds,
                        "ImageType",
                        []
                    )
                )

                if (
                    "LOCALIZER" in image_type
                    or "SCOUT" in image_type
                ):
                    continue

                uid = str(
                    getattr(
                        ds,
                        "SeriesInstanceUID",
                        ""
                    )
                )

                if target_uid and uid != target_uid:
                    continue

                groups.setdefault(
                    uid,
                    []
                ).append(p)

            except Exception:
                continue

        if not groups:
            raise RuntimeError(
                "没有找到可用CT Series"
            )

        # 如果输入目录包含多个series，默认选切片数最多的
        uid, files = max(
            groups.items(),
            key=lambda x: len(x[1])
        )

        if len(files) < 2:
            raise RuntimeError(
                "CT Series切片数量不足"
            )

        return uid, files

    # --------------------------------------------------------
    # DICOM Series → HU volume (x,y,z)
    # --------------------------------------------------------
    def _load_ct_volume(self, input_path):

        series_uid, files = self._find_series(
            input_path
        )

        records = []

        # 读取第一张方向
        first_meta = None

        for p in files:
            try:
                ds = pydicom.dcmread(
                    str(p),
                    stop_before_pixels=True,
                    force=True
                )

                if hasattr(
                    ds,
                    "ImageOrientationPatient"
                ):
                    first_meta = ds
                    break

            except Exception:
                continue

        normal = None

        if (
            first_meta is not None
            and hasattr(
                first_meta,
                "ImageOrientationPatient"
            )
        ):
            iop = np.asarray(
                first_meta.ImageOrientationPatient,
                dtype=float
            )

            if len(iop) == 6:
                row_cos = iop[:3]
                col_cos = iop[3:]
                normal = np.cross(
                    row_cos,
                    col_cos
                )

                n = np.linalg.norm(normal)

                if n > 0:
                    normal = normal / n

                    # BodyPartRegression为轴位CT模型；
                    # 明显非轴位重建不直接送入
                    if abs(float(normal[2])) < 0.80:
                        raise RuntimeError(
                            "当前Series明显不是轴位CT，"
                            "BodyPartRegression路由已阻止。"
                        )

        for p in files:

            try:
                ds = pydicom.dcmread(
                    str(p),
                    force=True
                )

                arr = ds.pixel_array.astype(
                    np.float32
                )

                slope = float(
                    getattr(
                        ds,
                        "RescaleSlope",
                        1.0
                    )
                )

                intercept = float(
                    getattr(
                        ds,
                        "RescaleIntercept",
                        0.0
                    )
                )

                hu = (
                    arr * slope
                    + intercept
                )

                pos = self._spatial_position(
                    ds,
                    normal
                )

                records.append(
                    (pos, hu, ds)
                )

            except Exception:
                continue

        if len(records) < 2:
            raise RuntimeError(
                "CT像素读取失败或有效切片不足"
            )

        records.sort(
            key=lambda x: x[0]
        )

        # 只保留一致矩阵大小
        shapes = {}

        for _, arr, _ in records:
            shapes[arr.shape] = (
                shapes.get(arr.shape, 0)
                + 1
            )

        main_shape = max(
            shapes.items(),
            key=lambda x: x[1]
        )[0]

        records = [
            x for x in records
            if x[1].shape == main_shape
        ]

        if len(records) < 2:
            raise RuntimeError(
                "有效CT切片不足"
            )

        ds0 = records[0][2]

        pixel_spacing = getattr(
            ds0,
            "PixelSpacing",
            None
        )

        if (
            pixel_spacing is None
            or len(pixel_spacing) < 2
        ):
            raise RuntimeError(
                "DICOM缺少PixelSpacing"
            )

        # DICOM PixelSpacing =
        # [row spacing, column spacing]
        y_spacing = float(
            pixel_spacing[0]
        )
        x_spacing = float(
            pixel_spacing[1]
        )

        positions = np.asarray(
            [x[0] for x in records],
            dtype=float
        )

        diffs = np.abs(
            np.diff(positions)
        )

        diffs = diffs[
            diffs > 1e-5
        ]

        if len(diffs):
            z_spacing = float(
                np.median(diffs)
            )
        else:
            z_spacing = float(
                getattr(
                    ds0,
                    "SpacingBetweenSlices",
                    getattr(
                        ds0,
                        "SliceThickness",
                        0
                    )
                )
            )

        if z_spacing <= 0:
            raise RuntimeError(
                "无法确定有效z-spacing"
            )

        # DICOM pixel array = (row=y, column=x)
        # BPR axis_ordering=(0,1,2)要求xyz
        # 因此每层转置成(x,y)，再沿z堆叠
        volume = np.stack(
            [
                arr.T
                for _, arr, _ in records
            ],
            axis=2
        ).astype(
            np.float32,
            copy=False
        )

        spacings = (
            x_spacing,
            y_spacing,
            z_spacing
        )

        return {
            "series_uid_internal": series_uid,
            "volume": volume,
            "pixel_spacings": spacings,
            "slice_count": len(records),
            "matrix": (
                int(main_shape[0]),
                int(main_shape[1])
            ),
        }

    # --------------------------------------------------------
    # 正式CT路由推理
    # --------------------------------------------------------
    def run(self, input_path):

        data = self._load_ct_volume(
            input_path
        )

        model = self._ensure_model()

        metadata = model.npy2json(
            data["volume"],
            output_path="",
            pixel_spacings=data[
                "pixel_spacings"
            ],
            axis_ordering=(0, 1, 2),
            ignore_invalid_z=False
        )

        # ----------------------------------------------------
        # BodyPartRegression的汇总tag在短范围CT中可能返回NONE，
        # 但region-level结果仍然可以明确指出身体区域。
        #
        # Phoenix保留原始tag用于审计，同时生成统一路由标签。
        # ----------------------------------------------------
        raw_tag = metadata.get(
            "body part examined tag"
        )

        body_regions = metadata.get(
            "body part examined"
        ) or {}

        canonical_order = [
            "legs",
            "pelvis",
            "abdomen",
            "chest",
            "shoulder-neck",
            "head",
        ]

        active_regions = []

        if isinstance(body_regions, dict):

            for region in canonical_order:

                values = body_regions.get(
                    region,
                    []
                )

                try:
                    has_values = (
                        values is not None
                        and len(values) > 0
                    )
                except Exception:
                    has_values = bool(values)

                if has_values:
                    active_regions.append(
                        region
                    )

        raw_text = str(
            raw_tag or ""
        ).strip()

        raw_upper = raw_text.upper()

        if (
            raw_text
            and raw_upper
            not in {
                "NONE",
                "UNKNOWN",
                "N/A",
                "NA",
            }
        ):
            normalized_tag = raw_text

        elif active_regions:

            # 单区域直接作为身体部位。
            if len(active_regions) == 1:
                normalized_tag = (
                    active_regions[0]
                )

            # 多区域CT保留完整覆盖范围。
            else:
                normalized_tag = (
                    " -> ".join(
                        active_regions
                    )
                )

        else:
            normalized_tag = (
                "UNDETERMINED"
            )

        chinese_names = {
            "legs": "下肢",
            "pelvis": "骨盆",
            "abdomen": "腹部",
            "chest": "胸部",
            "shoulder-neck": "肩颈部",
            "head": "头部",
        }

        if len(active_regions) == 1:
            body_part_display = (
                chinese_names.get(
                    active_regions[0],
                    active_regions[0]
                )
            )

        elif active_regions:
            body_part_display = (
                " → ".join(
                    chinese_names.get(
                        x,
                        x
                    )
                    for x in active_regions
                )
            )

        elif normalized_tag != "UNDETERMINED":
            body_part_display = normalized_tag

        else:
            body_part_display = (
                "部位未确定"
            )

        return {
            "model": "BodyPartRegression",
            "role": "CT身体部位路由",

            # Phoenix统一后的结果
            "body_part_examined_tag":
                normalized_tag,

            "body_part_display":
                body_part_display,

            "active_body_regions":
                active_regions,
            # 模型原始输出保留用于审计
            "body_part_examined_tag_raw":
                raw_tag,

            "body_part_examined":
                body_regions,
            "valid_z_spacing":
                metadata.get(
                    "valid z-spacing"
                ),
            "reverse_z_ordering":
                metadata.get(
                    "reverse z-ordering"
                ),
            "cleaned_slice_scores":
                metadata.get(
                    "cleaned slice scores"
                ),
            "slice_count":
                data["slice_count"],
            "matrix":
                list(data["matrix"]),
            "pixel_spacings_mm":
                list(
                    data["pixel_spacings"]
                ),
            "patient_identifiers_exported":
                False,
            "model_scope":
                "adult pelvis-to-head CT",
            "scope_exclusions": [
                "pediatric CT",
                "pregnancy",
                "legs"
            ],
        }

    def release(self):

        self.model = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass
