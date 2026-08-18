from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from core.model_adapter import ModelAdapter
from core.ct_nifti import series_to_nifti


class MonaiLungNoduleCTAdapter(ModelAdapter):

    name = "monai_lung_nodule_ct"

    def __init__(self):
        self.project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.bundle_root = (
            self.project_root
            / "04_AI模型"
            / "批量专家池"
            / "MONAI_CT专科"
            / "lung_nodule_ct_detection"
        )

        self.config = (
            self.bundle_root
            / "configs"
            / "inference.json"
        )

        self.weights = (
            self.bundle_root
            / "models"
            / "model.pt"
        )

        self.loaded = False

    def load(self):
        if not self.config.exists():
            raise RuntimeError(
                f"MONAI inference.json不存在: {self.config}"
            )

        if not self.weights.exists():
            raise RuntimeError(
                f"MONAI肺结节权重不存在: {self.weights}"
            )

        import monai

        self.loaded = True

    def unload(self):
        self.loaded = False

    def _select_ct_series(self, case):
        candidates = []

        for series in getattr(case, "series", []):
            modality = str(
                getattr(series, "modality", "")
            ).upper()

            files = getattr(
                series,
                "files",
                [],
            ) or []

            if modality == "CT" and len(files) >= 16:
                candidates.append(series)

        if not candidates:
            raise RuntimeError(
                "没有找到可用于肺结节检测的CT序列"
            )

        return max(
            candidates,
            key=lambda s: len(
                getattr(s, "files", []) or []
            ),
        )

    def _run_bundle(self, series, work_dir):
        input_nii = work_dir / "ct.nii.gz"

        series_to_nifti(
            series,
            input_nii,
        )

        datalist = work_dir / "dataset.json"

        datalist.write_text(
            json.dumps(
                {
                    "validation": [
                        {
                            "image": input_nii.name
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        output_dir = work_dir / "output"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_name = "phoenix_lung_result.json"

        cmd = [
            sys.executable,
            "-m",
            "monai.bundle",
            "run",

            "--config_file",
            str(self.config),

            "--bundle_root",
            str(self.bundle_root),

            "--dataset_dir",
            str(work_dir),

            "--data_list_file_path",
            str(datalist),

            "--output_dir",
            str(output_dir),

            "--output_filename",
            output_name,

            "--whether_raw_luna16",
            "true",

            "--amp",
            "false",

            "--dataloader#num_workers",
            "0",
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(self.bundle_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "MONAI肺结节模型运行失败:\n"
                + proc.stderr[-4000:]
            )

        result_file = (
            output_dir / output_name
        )

        if not result_file.exists():
            raise RuntimeError(
                "MONAI肺结节模型没有生成结果文件"
            )

        return json.loads(
            result_file.read_text(
                encoding="utf-8"
            )
        )

    def _prediction_items(self, data):
        if isinstance(data, list):
            return data

        if not isinstance(data, dict):
            return []

        for key in (
            "predictions",
            "results",
            "data",
        ):
            value = data.get(key)

            if isinstance(value, list):
                return value

        return [data]

    def _to_lesions(self, data):
        lesions = []

        for item in self._prediction_items(data):
            if not isinstance(item, dict):
                continue

            boxes = (
                item.get("box")
                or item.get("boxes")
                or []
            )

            scores = (
                item.get("label_scores")
                or item.get("scores")
                or []
            )

            labels = (
                item.get("label")
                or item.get("labels")
                or []
            )

            for index, box in enumerate(boxes):
                try:
                    score = float(
                        scores[index]
                    )
                except Exception:
                    score = None

                try:
                    label = int(
                        labels[index]
                    )
                except Exception:
                    label = 0

                lesions.append(
                    {
                        "type": "lung_nodule",
                        "finding": "肺结节候选灶",
                        "location": "肺",
                        "score": score,
                        "label": label,
                        "geometry": {
                            "box_3d": box,
                        },
                        "source": self.name,
                    }
                )

        return lesions

    def predict(self, case):
        if not self.loaded:
            raise RuntimeError(
                "monai_lung_nodule_ct 尚未load"
            )

        series = self._select_ct_series(case)

        # SimpleITK/NIfTI 在部分 Windows 环境下对中文输出路径兼容不稳定。
        # 临时推理目录固定使用项目 SSD 内的纯 ASCII 路径，
        # 因此网吧 D: 与医院 G: 均可自动适配。
        temp_root = (
            self.project_root
            / "08_temp_cache"
        )

        temp_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        work_dir = Path(
            tempfile.mkdtemp(
                prefix="monai_lung_",
                dir=str(temp_root),
            )
        )

        try:
            raw = self._run_bundle(
                series,
                work_dir,
            )

            lesions = self._to_lesions(raw)

            return {
                "processed_images": len(
                    getattr(series, "files", []) or []
                ),
                "lesions": lesions,
                "raw_prediction_count": len(lesions),
            }

        finally:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )
