from pathlib import Path
import gc
import re

import torch
import torch.nn as nn


MERLIN_ROOT = Path(
    r"D:\project_phoenix\04_AI模型\批量专家池\CT_通用\Merlin"
)


def _torch_load(path):
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )


def _unwrap_state(obj):
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Unsupported Merlin checkpoint: {type(obj)}"
        )

    for key in (
        "network_weights",
        "state_dict",
        "model_state_dict",
        "model",
    ):
        value = obj.get(key)
        if isinstance(value, dict):
            return value

    if obj and all(
        torch.is_tensor(v)
        for v in obj.values()
    ):
        return obj

    raise RuntimeError(
        "Cannot locate tensor state_dict "
        f"in keys={list(obj.keys())[:20]}"
    )


def _extract_component(state, prefix):
    out = {}

    if prefix:
        for k, v in state.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
    else:
        out = {
            k: v
            for k, v in state.items()
            if torch.is_tensor(v)
        }

    # 只保留 3D ResNet 主干。
    keep = {}

    for k, v in out.items():
        if (
            k.startswith("conv1.")
            or k.startswith("bn1.")
            or re.match(r"layer[1-4]\.", k)
        ):
            keep[k] = v

    if "conv1.weight" not in keep:
        raise RuntimeError(
            f"Merlin i3_resnet not found. "
            f"prefix={prefix!r}"
        )

    return keep


def _conv_from_weight(
    weight,
    stride=1,
    bias=False,
):
    kernel = tuple(weight.shape[2:])
    padding = tuple(k // 2 for k in kernel)

    return nn.Conv3d(
        in_channels=weight.shape[1],
        out_channels=weight.shape[0],
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=bias,
    )


class MerlinBottleneck3D(nn.Module):

    def __init__(
        self,
        state,
        prefix,
        stride,
    ):
        super().__init__()

        w1 = state[prefix + "conv1.weight"]
        w2 = state[prefix + "conv2.weight"]
        w3 = state[prefix + "conv3.weight"]

        self.conv1 = _conv_from_weight(
            w1,
            stride=1,
            bias=(prefix + "conv1.bias") in state,
        )

        self.bn1 = nn.BatchNorm3d(
            w1.shape[0]
        )

        self.conv2 = _conv_from_weight(
            w2,
            stride=stride,
            bias=(prefix + "conv2.bias") in state,
        )

        self.bn2 = nn.BatchNorm3d(
            w2.shape[0]
        )

        self.conv3 = _conv_from_weight(
            w3,
            stride=1,
            bias=(prefix + "conv3.bias") in state,
        )

        self.bn3 = nn.BatchNorm3d(
            w3.shape[0]
        )

        self.relu = nn.ReLU(inplace=True)

        down = prefix + "downsample.0.weight"

        if down in state:
            wd = state[down]

            self.downsample = nn.Sequential(
                _conv_from_weight(
                    wd,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(
                    wd.shape[0]
                ),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MerlinI3ResNetEncoder(nn.Module):

    def __init__(self, state):
        super().__init__()

        stem = state["conv1.weight"]

        self.conv1 = _conv_from_weight(
            stem,
            stride=(2, 2, 2),
            bias="conv1.bias" in state,
        )

        self.bn1 = nn.BatchNorm3d(
            stem.shape[0]
        )

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_stage(
            state, 1, stride=1
        )

        self.layer2 = self._make_stage(
            state, 2, stride=2
        )

        self.layer3 = self._make_stage(
            state, 3, stride=2
        )

        self.layer4 = self._make_stage(
            state, 4, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

    @staticmethod
    def _block_ids(state, stage):
        rx = re.compile(
            rf"^layer{stage}\.(\d+)\.conv1\.weight$"
        )

        ids = []

        for key in state:
            m = rx.match(key)

            if m:
                ids.append(int(m.group(1)))

        return sorted(set(ids))

    def _make_stage(
        self,
        state,
        stage,
        stride,
    ):
        blocks = []

        ids = self._block_ids(
            state,
            stage,
        )

        if not ids:
            raise RuntimeError(
                f"Merlin layer{stage} not found"
            )

        for block_id in ids:
            block_stride = (
                stride
                if block_id == ids[0]
                else 1
            )

            prefix = (
                f"layer{stage}."
                f"{block_id}."
            )

            blocks.append(
                MerlinBottleneck3D(
                    state,
                    prefix,
                    block_stride,
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature_map = x

        embedding = self.avgpool(x)
        embedding = torch.flatten(
            embedding,
            1,
        )

        return {
            "embedding": embedding,
            "feature_map": feature_map,
        }


class MerlinEncoderAdapter:

    def __init__(
        self,
        model_id,
        checkpoint,
        prefix,
    ):
        self.model_id = model_id
        self.checkpoint = Path(checkpoint)
        self.prefix = prefix

        self.task = "ct_3d_encoder"

        self.model = None

        self.status = (
            "ADAPTER_READY_UNTESTED"
        )

        self.load_report = None

    def validate_assets(self):
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                self.checkpoint
            )

        return True

    def load(self, device="cpu"):
        self.validate_assets()

        raw = _torch_load(
            self.checkpoint
        )

        state = _unwrap_state(raw)

        component = _extract_component(
            state,
            self.prefix,
        )

        model = MerlinI3ResNetEncoder(
            component
        )

        result = model.load_state_dict(
            component,
            strict=False,
        )

        missing = list(
            result.missing_keys
        )

        unexpected = list(
            result.unexpected_keys
        )

        self.load_report = {
            "component_tensors":
                len(component),

            "missing":
                missing,

            "unexpected":
                unexpected,
        }

        critical_missing = [
            k for k in missing
            if (
                k.startswith("conv1.")
                or k.startswith("bn1.")
                or k.startswith("layer")
            )
        ]

        if critical_missing:
            raise RuntimeError(
                "Merlin critical checkpoint "
                "mismatch: "
                + str(
                    critical_missing[:20]
                )
            )

        model.eval()
        model.to(device)

        self.model = model

        self.status = (
            "LOADED_UNTESTED"
        )

        del raw
        del state
        del component

        gc.collect()

        return self

    def run(self, volume):
        if self.model is None:
            raise RuntimeError(
                "Merlin is not loaded"
            )

        with torch.inference_mode():
            return self.model(volume)

    def unload(self):
        self.model = None

        self.status = (
            "ADAPTER_READY_UNTESTED"
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


MERLIN_CLIP_ENCODER = MerlinEncoderAdapter(
    model_id="merlin_clip_encoder",
    checkpoint=(
        MERLIN_ROOT
        / "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
    ),
    prefix="encode_image.i3_resnet.",
)


MERLIN_NNUNET_ENCODER = MerlinEncoderAdapter(
    model_id="merlin_nnunet_encoder",
    checkpoint=(
        MERLIN_ROOT
        / "nnUNetTrainerMerlin__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_best.pth"
    ),
    prefix="0.i3_resnet.",
)


MERLIN_RUNTIME_POOL = {
    "merlin_clip_encoder":
        MERLIN_CLIP_ENCODER,

    "merlin_nnunet_encoder":
        MERLIN_NNUNET_ENCODER,
}
