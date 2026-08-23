from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class ExpertInputAdapter:
    def _dicoms(self, path):
        import pydicom
        p = Path(path)
        files = [p] if p.is_file() else list(p.rglob("*"))
        out = []
        for f in files:
            if not f.is_file():
                continue
            try:
                out.append((f, pydicom.dcmread(str(f), force=True)))
            except Exception:
                pass
        return out

    def prepare_ct(self, path):
        items = []
        for _, ds in self._dicoms(path):
            if str(getattr(ds, "Modality", "")).upper() != "CT":
                continue
            a = ds.pixel_array.astype(np.float32)
            a *= float(getattr(ds, "RescaleSlope", 1))
            a += float(getattr(ds, "RescaleIntercept", 0))
            ipp = getattr(ds, "ImagePositionPatient", None)
            order = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else float(getattr(ds, "InstanceNumber", len(items)))
            items.append((order, a))
        if not items:
            raise RuntimeError("No CT DICOM")
        items.sort(key=lambda x: x[0])
        volume = np.stack([x[1] for x in items], axis=0)
        x = torch.from_numpy(volume).float()[None, None]
        x = torch.clamp(x, -1000, 1000)
        x = (x + 1000.0) / 2000.0
        return F.interpolate(x, size=(32, 256, 256), mode="trilinear", align_corners=False)

    def prepare_dr(self, path):
        modalities = {"DX", "DR", "CR", "XR", "MG"}
        for _, ds in self._dicoms(path):
            if str(getattr(ds, "Modality", "")).upper() not in modalities:
                continue
            a = ds.pixel_array.astype(np.float32)
            a -= a.min()
            m = a.max()
            if m > 0:
                a /= m
            return torch.from_numpy(a).float()[None, None]
        raise RuntimeError("No DR/DX DICOM")


EXPERT_INPUT_ADAPTER = ExpertInputAdapter()
