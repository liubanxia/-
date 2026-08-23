from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from core.ct_nifti import series_to_nifti
from core.ct_series_selector import select_ct_series
from core.model_adapter import ModelAdapter

LABELS = {1: "脑实质内出血", 2: "脑外出血", 3: "病灶周围水肿", 4: "脑室内出血"}


class BlastCTAdapter(ModelAdapter):
    name = "blast_ct_head"

    def __init__(self, cache_dir, temp_root=None):
        self.cache_dir = Path(cache_dir)
        self.temp_root = Path(temp_root) if temp_root else Path(tempfile.gettempdir()) / "project_phoenix"
        self.cli = None

    def load(self):
        cli = Path(sys.executable).parent / "blast-ct.exe"
        if not cli.exists():
            found = shutil.which("blast-ct")
            if not found: raise RuntimeError("未找到 blast-ct")
            cli = Path(found)
        if not self.cache_dir.exists(): raise FileNotFoundError(str(self.cache_dir))
        self.cli = cli

    def predict(self, case):
        if self.cli is None: raise RuntimeError("blast_ct_head 尚未load")
        series = select_ct_series(case, anatomy="head", minimum_images=16)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="phoenix_blast_", dir=str(self.temp_root)) as td:
            td = Path(td); input_nii = td / "ct.nii.gz"; output_nii = td / "blast.nii.gz"
            ordered = series_to_nifti(series, input_nii)
            env = os.environ.copy(); env["BLAST_CT_CACHE_DIR"] = str(self.cache_dir)
            run = subprocess.run([str(self.cli), "--input", str(input_nii), "--output", str(output_nii), "--device", "cpu"], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if run.returncode != 0 and not output_nii.exists(): raise RuntimeError("BLAST-CT运行失败: " + (run.stderr[-2000:] or run.stdout[-2000:] or f"returncode={run.returncode}"))
            if not output_nii.exists(): raise RuntimeError("BLAST-CT没有生成分割结果")
            seg = sitk.GetArrayFromImage(sitk.ReadImage(str(output_nii)))
            original_index = {str(Path(path).resolve()): index for index, path in enumerate(getattr(series, "files", []) or [])}
            lesions = []
            for class_id, label in LABELS.items():
                cc, count = ndimage.label(seg == class_id)
                for component_id in range(1, count + 1):
                    coords = np.argwhere(cc == component_id)
                    if len(coords) == 0: continue
                    z_values, z_counts = np.unique(coords[:, 0], return_counts=True); z = int(z_values[np.argmax(z_counts)])
                    if z < 0 or z >= len(ordered): continue
                    slice_coords = coords[coords[:, 0] == z]; y, x = np.round(slice_coords[:, 1:3].mean(axis=0)).astype(int)
                    file_path = Path(ordered[z]); image_index = original_index.get(str(file_path.resolve()), z)
                    lesions.append({"label": label, "finding": label, "confidence": 0.0, "series_uid": str(getattr(series, "series_uid", "")), "image_index": int(image_index), "point": (int(x), int(y)), "geometry_mode": "native_segmentation_pixel", "voxel_count": int(len(coords)), "source": self.name})
            lesions.sort(key=lambda item: item.get("voxel_count", 0), reverse=True)
            return {"model": self.name, "processed_images": len(ordered), "series_uid": str(getattr(series, "series_uid", "")), "lesions": lesions, "inference_backend": "blast_ct_cli", "device": "cpu", "warning": "BLAST-CT外部CLI返回非零状态；结果需复核。" if run.returncode != 0 else ""}

    def unload(self): self.cli = None
