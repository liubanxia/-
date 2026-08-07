import numpy as np


def get_pixel_array(dataset) -> np.ndarray:
    """
    从 DICOM Dataset 中读取原始像素矩阵。

    不修改原始 DICOM。
    """

    if not hasattr(dataset, "PixelData"):
        raise ValueError("DICOM 文件不存在 PixelData")

    return dataset.pixel_array


def ct_to_hu(dataset) -> np.ndarray:
    """
    将 CT 原始像素值转换为 HU。

    HU = PixelValue * RescaleSlope + RescaleIntercept
    """

    if getattr(dataset, "Modality", None) != "CT":
        raise ValueError("ct_to_hu 仅允许处理 CT")

    pixel_array = get_pixel_array(dataset).astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))

    hu_array = pixel_array * slope + intercept

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
        raise ValueError("WindowWidth 必须大于 0")

    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    clipped = np.clip(image, lower, upper)

    normalized = (
        (clipped - lower)
        / (upper - lower)
        * 255.0
    )

    return normalized.astype(np.uint8)


def normalize_dx(dataset) -> np.ndarray:
    """
    将 DX（DR）像素数据标准化到 8-bit 灰度范围。
    """

    if getattr(dataset, "Modality", None) != "DX":
        raise ValueError("normalize_dx 仅允许处理 DX")

    pixel_array = get_pixel_array(dataset).astype(np.float32)

    minimum = float(pixel_array.min())
    maximum = float(pixel_array.max())

    if maximum <= minimum:
        return np.zeros(
            pixel_array.shape,
            dtype=np.uint8,
        )

    normalized = (
        (pixel_array - minimum)
        / (maximum - minimum)
        * 255.0
    )

    image = normalized.astype(np.uint8)

    if getattr(
        dataset,
        "PhotometricInterpretation",
        "",
    ) == "MONOCHROME1":
        image = 255 - image

    return image