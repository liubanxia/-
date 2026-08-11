import math

import numpy as np

from dicom.pixel import apply_window, ct_to_hu


def prepare_windowed_ct_slice(
    dataset,
    window_center,
    window_width,
):
    """
    将单张CT DICOM转换为Phoenix统一AI基础输入。

    流程：
        DICOM
        -> 已验证HU
        -> 指定窗宽窗位
        -> float32
        -> 0.0 ~ 1.0

    当前阶段不做：
    - 图像缩放
    - 通道复制
    - ImageNet归一化
    - 模型特定预处理

    这些操作必须由未来具体模型配置决定，
    禁止提前猜测。
    """

    try:
        center = float(window_center)
        width = float(window_width)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "AI输入窗宽窗位必须是有效数值"
        ) from exc

    if not math.isfinite(center):
        raise ValueError(
            "AI输入WindowCenter必须是有限数值"
        )

    if not math.isfinite(width) or width <= 0:
        raise ValueError(
            "AI输入WindowWidth必须是大于0的有限数值"
        )

    hu_array = ct_to_hu(dataset)

    windowed = apply_window(
        hu_array,
        center,
        width,
    )

    normalized = windowed.astype(
        np.float32
    ) / 255.0

    if normalized.ndim != 2:
        raise ValueError(
            "Phoenix CT AI基础输入必须是二维图像"
        )

    if not np.isfinite(normalized).all():
        raise ValueError(
            "Phoenix CT AI基础输入包含非有限数值"
        )

    return normalized


def prepare_windowed_ct_triplet(
    datasets,
    window_center,
    window_width,
):
    """
    将明确指定的3张CT切片转换为Phoenix 2.5D基础输入。

    输入顺序固定：
        [前一层, 当前层, 后一层]

    返回：
        float32 NumPy数组
        形状：(3, H, W)
        数值范围：0.0 ~ 1.0

    注意：
    - 本函数不负责寻找相邻层；
    - 不根据InstanceNumber猜层面；
    - 不假定层间距恒定；
    - 三张切片必须已经由上层CT空间排序逻辑确认；
    - 当前不进行模型特定resize或归一化。
    """

    if not isinstance(datasets, (list, tuple)):
        raise TypeError(
            "2.5D输入必须提供3张CT dataset"
        )

    if len(datasets) != 3:
        raise ValueError(
            "2.5D输入必须严格包含3张CT切片"
        )

    prepared_slices = [
        prepare_windowed_ct_slice(
            dataset=dataset,
            window_center=window_center,
            window_width=window_width,
        )
        for dataset in datasets
    ]

    reference_shape = prepared_slices[0].shape

    for image in prepared_slices[1:]:
        if image.shape != reference_shape:
            raise ValueError(
                "2.5D三张CT切片尺寸不一致"
            )

    stacked = np.stack(
        prepared_slices,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(stacked).all():
        raise ValueError(
            "Phoenix 2.5D CT输入包含非有限数值"
        )

    return stacked
