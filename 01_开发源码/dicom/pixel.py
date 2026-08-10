import math

import numpy as np


def get_pixel_array(dataset) -> np.ndarray:
    """
    安全读取二维灰阶 DICOM 像素矩阵。

    当前 V1 仅支持二维单帧灰阶影像。
    """

    if not hasattr(dataset, "PixelData"):
        raise ValueError(
            "DICOM 文件缺失 PixelData"
        )

    try:
        pixel_array = dataset.pixel_array
    except Exception as exc:
        raise ValueError(
            "DICOM 像素数据解码失败"
        ) from exc

    if pixel_array.ndim != 2:
        raise ValueError(
            "当前仅支持二维灰阶影像，"
            f"实际维度: {pixel_array.shape}"
        )

    return pixel_array


def _get_required_ct_float(
    dataset,
    attribute_name: str,
) -> float:
    """
    读取 CT 必需的数值型 DICOM 属性。

    缺失、无法转换或不是有限数值时安全停止。
    """

    if not hasattr(dataset, attribute_name):
        raise ValueError(
            f"CT 缺失 {attribute_name}"
        )

    try:
        value = float(
            getattr(dataset, attribute_name)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CT {attribute_name} 数值异常"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"CT {attribute_name} 数值异常"
        )

    return value


def validate_ct_pixel_dataset(dataset):
    """
    校验单张 CT 的基础像素结构及 HU 标定信息。

    本函数只验证当前 V1 阅片所必需的信息，
    不修改原始 DICOM。
    """

    if getattr(dataset, "Modality", None) != "CT":
        raise ValueError(
            "CT 像素校验仅允许处理 CT"
        )

    pixel_array = get_pixel_array(dataset)

    # --------------------------------------------------
    # Rows / Columns
    # --------------------------------------------------
    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
    except Exception as exc:
        raise ValueError(
            "CT 缺失或无法读取 Rows / Columns"
        ) from exc

    if rows <= 0 or columns <= 0:
        raise ValueError(
            "CT Rows / Columns 数值异常"
        )

    if pixel_array.shape != (rows, columns):
        raise ValueError(
            "CT 像素矩阵尺寸与 Rows / Columns 不一致"
        )

    # --------------------------------------------------
    # 单通道灰阶
    # --------------------------------------------------
    try:
        samples_per_pixel = int(
            dataset.SamplesPerPixel
        )
    except Exception as exc:
        raise ValueError(
            "CT 缺失或无法读取 SamplesPerPixel"
        ) from exc

    if samples_per_pixel != 1:
        raise ValueError(
            "当前仅支持单通道 CT 灰阶影像"
        )

    photometric = str(
        getattr(
            dataset,
            "PhotometricInterpretation",
            "",
        )
    ).strip().upper()

    if photometric not in (
        "MONOCHROME1",
        "MONOCHROME2",
    ):
        raise ValueError(
            "CT PhotometricInterpretation 不受支持"
        )

    # --------------------------------------------------
    # 像素位深基础一致性
    # --------------------------------------------------
    required_integer_attributes = (
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
    )

    integer_values = {}

    for attribute_name in required_integer_attributes:
        if not hasattr(dataset, attribute_name):
            raise ValueError(
                f"CT 缺失 {attribute_name}"
            )

        try:
            integer_values[attribute_name] = int(
                getattr(dataset, attribute_name)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CT {attribute_name} 数值异常"
            ) from exc

    bits_allocated = integer_values[
        "BitsAllocated"
    ]
    bits_stored = integer_values[
        "BitsStored"
    ]
    high_bit = integer_values[
        "HighBit"
    ]
    pixel_representation = integer_values[
        "PixelRepresentation"
    ]

    if (
        bits_allocated <= 0
        or bits_stored <= 0
        or bits_stored > bits_allocated
    ):
        raise ValueError(
            "CT 像素位深信息异常"
        )

    if high_bit != bits_stored - 1:
        raise ValueError(
            "CT HighBit 与 BitsStored 不一致"
        )

    if pixel_representation not in (0, 1):
        raise ValueError(
            "CT PixelRepresentation 数值异常"
        )

    # --------------------------------------------------
    # HU 标定
    # 禁止缺失时自动猜测 1 / 0
    # --------------------------------------------------
    slope = _get_required_ct_float(
        dataset,
        "RescaleSlope",
    )

    intercept = _get_required_ct_float(
        dataset,
        "RescaleIntercept",
    )

    return pixel_array, slope, intercept


def ct_to_hu(dataset) -> np.ndarray:
    """
    将 CT 原始像素值转换为 HU。

    HU = PixelValue * RescaleSlope + RescaleIntercept

    RescaleSlope / RescaleIntercept 缺失或异常时，
    不使用默认值猜测，直接安全停止。
    """

    (
        pixel_array,
        slope,
        intercept,
    ) = validate_ct_pixel_dataset(dataset)

    pixel_array = pixel_array.astype(
        np.float32
    )

    hu_array = (
        pixel_array * slope
        + intercept
    )

    return hu_array


def apply_window(
    image: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    应用窗宽窗位，并转换成 8-bit 灰度图。

    返回范围：
    0 ~ 255
    """

    if window_width <= 0:
        raise ValueError(
            "WindowWidth 必须大于 0"
        )

    lower = (
        window_center
        - window_width / 2
    )
    upper = (
        window_center
        + window_width / 2
    )

    clipped = np.clip(
        image,
        lower,
        upper,
    )

    normalized = (
        (clipped - lower)
        / (upper - lower)
        * 255.0
    )

    return normalized.astype(
        np.uint8
    )


def normalize_dx(dataset) -> np.ndarray:
    """
    将 DX（DR）像素数据标准化到 8-bit 灰度范围。
    """

    if getattr(dataset, "Modality", None) != "DX":
        raise ValueError(
            "normalize_dx 仅允许处理 DX"
        )

    pixel_array = get_pixel_array(
        dataset
    ).astype(np.float32)

    minimum = float(
        pixel_array.min()
    )
    maximum = float(
        pixel_array.max()
    )

    if maximum <= minimum:
        raise ValueError(
            "DX 像素值范围无效，无法显示影像"
        )

    normalized = (
        (pixel_array - minimum)
        / (maximum - minimum)
        * 255.0
    )

    image = normalized.astype(
        np.uint8
    )

    photometric = str(
        getattr(
            dataset,
            "PhotometricInterpretation",
            "",
        )
    ).strip().upper()

    if photometric == "MONOCHROME1":
        image = 255 - image

    elif photometric != "MONOCHROME2":
        raise ValueError(
            "DX PhotometricInterpretation 不受支持"
        )

    return image
