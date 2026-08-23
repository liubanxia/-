from pathlib import Path
import gc
import re
import torch
import torch.nn as nn


def _torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _unwrap_state(obj):
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported Merlin checkpoint: {type(obj)}")
    for key in ("network_weights", "state_dict", "model_state_dict", "model"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise RuntimeError("Cannot locate tensor state_dict")


def _extract_component(state, prefix):
    out = (
        {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if prefix
        else {k: v for k, v in state.items() if torch.is_tensor(v)}
    )
    keep = {
        k: v
        for k, v in out.items()
        if k.startswith("conv1.")
        or k.startswith("bn1.")
        or re.match(r"layer[1-4]\.", k)
    }
    if "conv1.weight" not in keep:
        raise RuntimeError(f"Merlin i3_resnet not found. prefix={prefix!r}")
    return keep


def _conv(weight, stride=1, bias=False):
    kernel = tuple(weight.shape[2:])
    padding = tuple(k // 2 for k in kernel)
    return nn.Conv3d(
        weight.shape[1],
        weight.shape[0],
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=bias,
    )


class MerlinBottleneck3D(nn.Module):
    def __init__(self, state, prefix, stride):
        super().__init__()
        w1 = state[prefix + "conv1.weight"]
        w2 = state[prefix + "conv2.weight"]
        w3 = state[prefix + "conv3.weight"]
        self.conv1 = _conv(w1)
        self.bn1 = nn.BatchNorm3d(w1.shape[0])
        self.conv2 = _conv(w2, stride=stride)
        self.bn2 = nn.BatchNorm3d(w2.shape[0])
        self.conv3 = _conv(w3)
        self.bn3 = nn.BatchNorm3d(w3.shape[0])
        self.relu = nn.ReLU(inplace=True)
        down = prefix + "downsample.0.weight"
        self.downsample = (
            nn.Sequential(
                _conv(state[down], stride=stride),
                nn.BatchNorm3d(state[down].shape[0]),
            )
            if down in state
            else None
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class MerlinI3ResNetEncoder(nn.Module):
    def __init__(self, state):
        super().__init__()
        stem = state["conv1.weight"]
        self.conv1 = _conv(stem, stride=(2, 2, 2))
        self.bn1 = nn.BatchNorm3d(stem.shape[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._stage(state, 1, 1)
        self.layer2 = self._stage(state, 2, 2)
        self.layer3 = self._stage(state, 3, 2)
        self.layer4 = self._stage(state, 4, 2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    @staticmethod
    def _ids(state, stage):
        rx = re.compile(rf"^layer{stage}\.(\d+)\.conv1\.weight$")
        ids = []
        for key in state:
            match = rx.match(key)
            if match:
                ids.append(int(match.group(1)))
        return sorted(set(ids))

    def _stage(self, state, stage, stride):
        ids = self._ids(state, stage)
        if not ids:
            raise RuntimeError(f"Merlin layer{stage} not found")
        return nn.Sequential(
            *[
                MerlinBottleneck3D(
                    state,
                    f"layer{stage}.{block_id}.",
                    stride if block_id == ids[0] else 1,
                )
                for block_id in ids
            ]
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return {
            "embedding": torch.flatten(self.avgpool(x), 1),
            "feature_map": x,
        }


class MerlinEncoderAdapter:
    def __init__(self, model_id, checkpoint, prefix=""):
        self.model_id = model_id
        self.checkpoint = Path(checkpoint)
        self.prefix = prefix
        self.task = "ct_3d_encoder"
        self.model = None
        self.status = "ADAPTER_READY_UNTESTED"

    def load(self, device="cpu"):
        if not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        raw = _torch_load(self.checkpoint)
        component = _extract_component(_unwrap_state(raw), self.prefix)
        model = MerlinI3ResNetEncoder(component)
        result = model.load_state_dict(component, strict=False)
        critical = [
            key
            for key in result.missing_keys
            if key.startswith(("conv1.", "bn1.", "layer"))
        ]
        if critical:
            raise RuntimeError("Merlin critical checkpoint mismatch: " + str(critical[:20]))
        model.eval().to(device)
        self.model = model
        self.status = "LOADED_UNTESTED"
        gc.collect()
        return self

    def run(self, volume):
        if self.model is None:
            raise RuntimeError("Merlin is not loaded")
        with torch.inference_mode():
            return self.model(volume)

    def unload(self):
        self.model = None
        self.status = "ADAPTER_READY_UNTESTED"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_merlin_runtime_pool(merlin_root):
    root = Path(merlin_root)
    return {
        "merlin_clip_encoder": MerlinEncoderAdapter(
            "merlin_clip_encoder",
            root / "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
            "encode_image.i3_resnet.",
        ),
        "merlin_nnunet_encoder": MerlinEncoderAdapter(
            "merlin_nnunet_encoder",
            root / "nnUNetTrainerMerlin__nnUNetPlans__3d_fullres" / "fold_0" / "checkpoint_best.pth",
            "0.i3_resnet.",
        ),
    }


MERLIN_RUNTIME_POOL = {}
