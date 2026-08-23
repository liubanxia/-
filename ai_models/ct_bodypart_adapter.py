from pathlib import Path
import gc
import os
import sys

import numpy as np
import pydicom


class BodyPartRegressionCTAdapter:
    """DICOM-to-BodyPartRegression CT adapter without patient identifiers."""

    def __init__(self, project_root, repo_path, model_dir):
        self.project_root = Path(project_root)
        self.repo_path = Path(repo_path)
        self.model_dir = Path(model_dir)
        self.model = None

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        repo = str(self.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from bpreg.inference.inference_model import InferenceModel
        base_dir = str(self.model_dir)
        if not base_dir.endswith(("/", "\\")):
            base_dir += os.sep
        self.model = InferenceModel(base_dir=base_dir, gpu=False, warning_to_error=True)
        return self.model

    @staticmethod
    def _spatial_position(ds, normal=None):
        try:
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            return float(np.dot(ipp, normal)) if normal is not None else float(ipp[2])
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

    def _find_series(self, input_path):
        input_path = Path(input_path)
        target_uid = None
        if input_path.is_file():
            first = pydicom.dcmread(str(input_path), stop_before_pixels=True, force=True)
            if str(getattr(first, "Modality", "")).upper() != "CT":
                raise ValueError("输入文件不是CT DICOM")
            target_uid = str(getattr(first, "SeriesInstanceUID", ""))
            scan_root = input_path.parent
        else:
            scan_root = input_path

        groups = {}
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                if str(getattr(ds, "Modality", "")).upper() != "CT":
                    continue
                image_type = "\\".join(str(x).upper() for x in getattr(ds, "ImageType", []))
                if "LOCALIZER" in image_type or "SCOUT" in image_type:
                    continue
                uid = str(getattr(ds, "SeriesInstanceUID", ""))
                if target_uid and uid != target_uid:
                    continue
                groups.setdefault(uid, []).append(path)
            except Exception:
                continue

        if not groups:
            raise RuntimeError("没有找到可用CT Series")
        uid, files = max(groups.items(), key=lambda item: len(item[1]))
        if len(files) < 2:
            raise RuntimeError("CT Series切片数量不足")
        return uid, files

    def _load_ct_volume(self, input_path):
        series_uid, files = self._find_series(input_path)
        first_meta = None
        for path in files:
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                if hasattr(ds, "ImageOrientationPatient"):
                    first_meta = ds
                    break
            except Exception:
                continue

        normal = None
        if first_meta is not None and hasattr(first_meta, "ImageOrientationPatient"):
            iop = np.asarray(first_meta.ImageOrientationPatient, dtype=float)
            if len(iop) == 6:
                normal = np.cross(iop[:3], iop[3:])
                length = np.linalg.norm(normal)
                if length > 0:
                    normal = normal / length
                    if abs(float(normal[2])) < 0.80:
                        raise RuntimeError("当前Series明显不是轴位CT，BodyPartRegression路由已阻止。")

        records = []
        for path in files:
            try:
                ds = pydicom.dcmread(str(path), force=True)
                arr = ds.pixel_array.astype(np.float32)
                hu = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
                records.append((self._spatial_position(ds, normal), hu, ds))
            except Exception:
                continue

        if len(records) < 2:
            raise RuntimeError("CT像素读取失败或有效切片不足")
        records.sort(key=lambda item: item[0])

        shapes = {}
        for _, arr, _ in records:
            shapes[arr.shape] = shapes.get(arr.shape, 0) + 1
        main_shape = max(shapes.items(), key=lambda item: item[1])[0]
        records = [item for item in records if item[1].shape == main_shape]
        if len(records) < 2:
            raise RuntimeError("有效CT切片不足")

        ds0 = records[0][2]
        spacing = getattr(ds0, "PixelSpacing", None)
        if spacing is None or len(spacing) < 2:
            raise RuntimeError("DICOM缺少PixelSpacing")
        y_spacing, x_spacing = float(spacing[0]), float(spacing[1])
        positions = np.asarray([item[0] for item in records], dtype=float)
        diffs = np.abs(np.diff(positions))
        diffs = diffs[diffs > 1e-5]
        z_spacing = float(np.median(diffs)) if len(diffs) else float(getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", 0)))
        if z_spacing <= 0:
            raise RuntimeError("无法确定有效z-spacing")

        volume = np.stack([arr.T for _, arr, _ in records], axis=2).astype(np.float32, copy=False)
        return {
            "series_uid_internal": series_uid,
            "volume": volume,
            "pixel_spacings": (x_spacing, y_spacing, z_spacing),
            "slice_count": len(records),
            "matrix": (int(main_shape[0]), int(main_shape[1])),
        }

    def run(self, input_path):
        data = self._load_ct_volume(input_path)
        metadata = self._ensure_model().npy2json(
            data["volume"], output_path="", pixel_spacings=data["pixel_spacings"],
            axis_ordering=(0, 1, 2), ignore_invalid_z=False,
        )
        raw_tag = metadata.get("body part examined tag")
        body_regions = metadata.get("body part examined") or {}
        canonical = ["legs", "pelvis", "abdomen", "chest", "shoulder-neck", "head"]
        active = []
        if isinstance(body_regions, dict):
            for region in canonical:
                values = body_regions.get(region, [])
                try:
                    present = values is not None and len(values) > 0
                except Exception:
                    present = bool(values)
                if present:
                    active.append(region)

        raw_text = str(raw_tag or "").strip()
        if raw_text and raw_text.upper() not in {"NONE", "UNKNOWN", "N/A", "NA"}:
            normalized = raw_text
        elif active:
            normalized = active[0] if len(active) == 1 else " -> ".join(active)
        else:
            normalized = "UNDETERMINED"

        names = {"legs": "下肢", "pelvis": "骨盆", "abdomen": "腹部", "chest": "胸部", "shoulder-neck": "肩颈部", "head": "头部"}
        display = names.get(active[0], active[0]) if len(active) == 1 else " → ".join(names.get(x, x) for x in active) if active else normalized if normalized != "UNDETERMINED" else "部位未确定"

        return {
            "model": "BodyPartRegression",
            "role": "CT身体部位路由",
            "body_part_examined_tag": normalized,
            "body_part_display": display,
            "active_body_regions": active,
            "body_part_examined_tag_raw": raw_tag,
            "body_part_examined": body_regions,
            "valid_z_spacing": metadata.get("valid z-spacing"),
            "reverse_z_ordering": metadata.get("reverse z-ordering"),
            "cleaned_slice_scores": metadata.get("cleaned slice scores"),
            "slice_count": data["slice_count"],
            "matrix": list(data["matrix"]),
            "pixel_spacings_mm": list(data["pixel_spacings"]),
            "patient_identifiers_exported": False,
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
