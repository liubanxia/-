import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from core.model_adapter import ModelAdapter
from core.ct_nifti import series_to_nifti


LABELS = {
    1: "脑实质内出血",
    2: "脑外出血",
    3: "病灶周围水肿",
    4: "脑室内出血",
}


class BlastCTAdapter(ModelAdapter):

    name = "blast_ct_head"

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cli = None

    def load(self):
        cli = Path(sys.executable).parent / "blast-ct.exe"

        if not cli.exists():
            found = shutil.which("blast-ct")
            if not found:
                raise RuntimeError("未找到 blast-ct")
            cli = Path(found)

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                str(self.cache_dir)
            )

        self.cli = cli

    def predict(self, case):
        ct_series = [
            x for x in case.series
            if str(x.modality).upper() == "CT"
        ]

        if not ct_series:
            return {
                "model": self.name,
                "error": "没有CT序列",
            }

        series = max(
            ct_series,
            key=lambda x: len(x.files),
        )

        with tempfile.TemporaryDirectory(
            prefix="phoenix_blast_"
        ) as td:
            td = Path(td)

            input_nii = td / "ct.nii.gz"
            output_nii = td / "blast.nii.gz"

            ordered = series_to_nifti(
                series,
                input_nii,
            )

            env = os.environ.copy()
            env["BLAST_CT_CACHE_DIR"] = str(
                self.cache_dir
            )

            cmd = [
                str(self.cli),
                "--input", str(input_nii),
                "--output", str(output_nii),
                "--device", "cpu",
            ]

            run = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
            )

            cleanup_warning = ""

            if run.returncode != 0:
                if output_nii.exists():
                    cleanup_warning = (
                        "BLAST-CT预测已完成；Windows清理临时日志失败"
                    )
                else:
                    return {
                        "model": self.name,
                        "error": (
                            run.stderr[-1500:]
                            or run.stdout[-1500:]
                        ),
                    }

            seg = sitk.GetArrayFromImage(
                sitk.ReadImage(str(output_nii))
            )

            lesions = []

            for class_id, label in LABELS.items():
                cc, count = ndimage.label(
                    seg == class_id
                )

                for n in range(1, count + 1):
                    coords = np.argwhere(cc == n)

                    if len(coords) == 0:
                        continue

                    z, y, x = np.round(
                        coords.mean(axis=0)
                    ).astype(int)

                    if z >= len(ordered):
                        continue

                    file_path = ordered[z]

                    try:
                        image_index = series.files.index(
                            file_path
                        )
                    except ValueError:
                        image_index = int(z)

                    voxel_count = int(len(coords))

                    lesions.append({
                        "label": label,
                        "confidence": 0.0,
                        "series_uid": series.series_uid,
                        "image_index": image_index,
                        "point": (int(x), int(y)),
                        "voxel_count": voxel_count,
                    })

            return {
                "model": self.name,
                "processed_images": len(ordered),
                "lesions": lesions,
            }

    def unload(self):
        self.cli = None
