import numpy as np


MODEL_INPUT_SIZE = 1024
MODEL_INPUT_NAME = "images"


WRIST_KEYWORDS = (
    "WRIST",
    "CARPAL",
    "腕",
    "腕关节",
)


def _dicom_patient_age_to_years(patient_age):
    """
    解析DICOM PatientAge（AS VR）。
    示例：005Y / 006M / 003W / 010D。
    """

    value = str(patient_age).strip().upper()

    if len(value) != 4:
        raise ValueError(
            "视觉B PatientAge格式无效"
        )

    number_text = value[:3]
    unit = value[3]

    if not number_text.isdigit():
        raise ValueError(
            "视觉B PatientAge数值无效"
        )

    number = int(number_text)

    if unit == "Y":
        return float(number)

    if unit == "M":
        return number / 12.0

    if unit == "W":
        return number / 52.0

    if unit == "D":
        return number / 365.25

    raise ValueError(
        "视觉B PatientAge单位无效"
    )


def validate_yolov8_rescbam_domain(
    series_context,
):
    """
    验证YOLOv8_ResCBAM的适用域。

    当前真实模型仅允许：
    - Modality明确为DX；
    - DICOM部位/检查描述中明确出现腕部信息。

    不允许根据图像内容自动猜测适用部位。
    """

    if not isinstance(
        series_context,
        dict,
    ):
        raise TypeError(
            "视觉B series_context必须是dict"
        )

    modality = str(
        series_context.get(
            "modality",
            "",
        )
    ).strip().upper()

    if modality != "DX":
        raise ValueError(
            "YOLOv8_ResCBAM仅允许用于DX腕部影像"
        )

    body_part_examined = str(
        series_context.get(
            "body_part_examined",
            "",
        )
    ).strip()

    study_description = str(
        series_context.get(
            "study_description",
            "",
        )
    ).strip()

    series_description = str(
        series_context.get(
            "series_description",
            "",
        )
    ).strip()

    searchable_text = " ".join(
        (
            body_part_examined,
            study_description,
            series_description,
        )
    ).upper()

    is_wrist = any(
        keyword.upper() in searchable_text
        for keyword in WRIST_KEYWORDS
    )

    if not is_wrist:
        raise ValueError(
            "YOLOv8_ResCBAM拒绝推理："
            "当前DICOM未明确标识为腕部DX影像"
        )

    patient_age = str(
        series_context.get(
            "patient_age",
            "",
        )
    ).strip().upper()

    if not patient_age:
        raise ValueError(
            "YOLOv8_ResCBAM拒绝推理："
            "当前DICOM缺少PatientAge，"
            "无法确认儿童适用域"
        )

    age_years = _dicom_patient_age_to_years(
        patient_age
    )

    # Phoenix当前工程安全边界：
    # 仅允许明确小于18岁的病例进入该儿童腕部模型。
    if age_years >= 18.0:
        raise ValueError(
            "YOLOv8_ResCBAM拒绝推理："
            "当前病例不属于Phoenix定义的儿童适用域"
        )

    return True


def build_yolov8_rescbam_input(series_context):
    """
    为 YOLOv8_ResCBAM 构建 ONNX 输入。

    输入要求：
    - series_context["current_image_array"]：
      当前二维 8-bit 灰阶影像，坐标对应原始 DICOM 像素。

    预处理：
    1. 保持宽高比缩放；
    2. LetterBox 到 1024 x 1024；
    3. 灰阶复制为 3 通道；
    4. uint8 -> float32；
    5. /255.0；
    6. HWC -> NCHW；
    7. batch=1。

    本函数同时把几何反变换参数写入 series_context，
    供后续 decoder 将模型 bbox 恢复到原始 DICOM 坐标。
    """

    validate_yolov8_rescbam_domain(
        series_context
    )

    image = series_context.get(
        "current_image_array"
    )

    if image is None:
        raise ValueError(
            "视觉B缺少current_image_array"
        )

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            "YOLOv8_ResCBAM仅接受二维灰阶影像"
        )

    if image.dtype != np.uint8:
        raise TypeError(
            "current_image_array必须是uint8"
        )

    original_height, original_width = image.shape

    if original_height <= 0 or original_width <= 0:
        raise ValueError(
            "视觉B原始影像尺寸无效"
        )

    target = MODEL_INPUT_SIZE

    ratio = min(
        target / original_height,
        target / original_width,
    )

    resized_width = int(
        round(original_width * ratio)
    )
    resized_height = int(
        round(original_height * ratio)
    )

    if resized_width <= 0 or resized_height <= 0:
        raise ValueError(
            "视觉B LetterBox缩放结果无效"
        )

    # 使用 OpenCV，与作者 Ultralytics 的 LetterBox 行为保持一致。
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "视觉B预处理缺少opencv-python"
        ) from exc

    if (
        resized_width != original_width
        or resized_height != original_height
    ):
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        resized = image.copy()

    dw = target - resized_width
    dh = target - resized_height

    dw_half = dw / 2.0
    dh_half = dh / 2.0

    left = int(round(dw_half - 0.1))
    right = int(round(dw_half + 0.1))
    top = int(round(dh_half - 0.1))
    bottom = int(round(dh_half + 0.1))

    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=114,
    )

    if letterboxed.shape != (
        target,
        target,
    ):
        raise RuntimeError(
            "视觉B LetterBox最终尺寸不是1024x1024"
        )

    # 灰阶复制为3通道。
    image_3ch = np.repeat(
        letterboxed[:, :, None],
        3,
        axis=2,
    )

    # HWC -> CHW，uint8 -> float32，归一化到0~1。
    tensor = (
        image_3ch
        .transpose(2, 0, 1)
        .astype(np.float32)
        / 255.0
    )

    tensor = np.ascontiguousarray(
        tensor[None, ...]
    )

    if tensor.shape != (
        1,
        3,
        target,
        target,
    ):
        raise RuntimeError(
            "视觉B ONNX输入shape异常"
        )

    # 保存后处理需要的几何信息。
    # 注意：这是当前单次推理上下文，不写入全局状态。
    series_context[
        "visual_b_preprocess"
    ] = {
        "model_input_size": target,
        "original_width": int(original_width),
        "original_height": int(original_height),
        "ratio": float(ratio),
        "pad_left": int(left),
        "pad_top": int(top),
        "pad_right": int(right),
        "pad_bottom": int(bottom),
    }

    return {
        MODEL_INPUT_NAME: tensor
    }


FRACTURE_CLASS_ID = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


def _xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)

    result = np.empty_like(boxes)

    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

    return result


def _box_iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)

    intersection = intersection_w * intersection_h

    box_area = max(
        0.0,
        float(box[2] - box[0]),
    ) * max(
        0.0,
        float(box[3] - box[1]),
    )

    boxes_area = (
        np.maximum(
            0.0,
            boxes[:, 2] - boxes[:, 0],
        )
        * np.maximum(
            0.0,
            boxes[:, 3] - boxes[:, 1],
        )
    )

    union = box_area + boxes_area - intersection

    iou = np.zeros_like(
        intersection,
        dtype=np.float32,
    )

    valid = union > 0.0

    iou[valid] = (
        intersection[valid]
        / union[valid]
    )

    return iou


def _nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]

    keep = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        remaining = order[1:]

        ious = _box_iou_one_to_many(
            boxes[current],
            boxes[remaining],
        )

        order = remaining[
            ious <= iou_threshold
        ]

    return keep


def decode_yolov8_rescbam_fractures(
    raw_outputs,
    series_context,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
):
    """
    解析 YOLOv8_ResCBAM ONNX 输出。

    模型输出：
        output0 shape = [1, 13, 21504]

    13个通道：
        0~3   = xywh
        4~12  = 9个类别概率（模型内部已sigmoid）

    当前视觉B仅提取：
        class_id = 3
        class_name = fracture

    返回：
        VisualBOutputParser 可消费的候选草稿列表。

    bbox最终转换回原始DICOM像素坐标。
    """

    if not isinstance(
        series_context,
        dict,
    ):
        raise TypeError(
            "视觉B series_context必须是dict"
        )

    preprocess = series_context.get(
        "visual_b_preprocess"
    )

    if not isinstance(
        preprocess,
        dict,
    ):
        raise ValueError(
            "视觉B缺少visual_b_preprocess几何信息"
        )

    if not isinstance(
        raw_outputs,
        (list, tuple),
    ):
        raise TypeError(
            "视觉B raw_outputs必须是list或tuple"
        )

    if len(raw_outputs) != 1:
        raise ValueError(
            "YOLOv8_ResCBAM必须只有一个ONNX输出"
        )

    prediction = np.asarray(
        raw_outputs[0],
        dtype=np.float32,
    )

    if prediction.shape != (
        1,
        13,
        21504,
    ):
        raise ValueError(
            "YOLOv8_ResCBAM输出shape异常："
            f"{prediction.shape}"
        )

    # [1,13,21504] -> [21504,13]
    prediction = prediction[0].T

    boxes_xywh = prediction[:, 0:4]

    # class_id=3，对应输出通道：
    # 4 + 3 = 7
    fracture_scores = prediction[
        :,
        4 + FRACTURE_CLASS_ID,
    ]

    valid = np.isfinite(
        fracture_scores
    )

    valid &= (
        fracture_scores
        >= float(confidence_threshold)
    )

    if not np.any(valid):
        return []

    boxes_xywh = boxes_xywh[valid]
    scores = fracture_scores[valid]

    finite_boxes = np.isfinite(
        boxes_xywh
    ).all(axis=1)

    boxes_xywh = boxes_xywh[
        finite_boxes
    ]
    scores = scores[
        finite_boxes
    ]

    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = _xywh_to_xyxy(
        boxes_xywh
    )

    keep = _nms(
        boxes_xyxy,
        scores,
        float(iou_threshold),
    )

    ratio = float(
        preprocess["ratio"]
    )
    pad_left = float(
        preprocess["pad_left"]
    )
    pad_top = float(
        preprocess["pad_top"]
    )

    original_width = int(
        preprocess["original_width"]
    )
    original_height = int(
        preprocess["original_height"]
    )

    if ratio <= 0.0:
        raise ValueError(
            "视觉B LetterBox ratio无效"
        )

    slice_index = int(
        series_context.get(
            "current_slice_index",
            0,
        )
    )

    candidates = []

    for index in keep:
        x1, y1, x2, y2 = (
            boxes_xyxy[index]
        )

        # 撤销LetterBox padding。
        x1 = (x1 - pad_left) / ratio
        x2 = (x2 - pad_left) / ratio
        y1 = (y1 - pad_top) / ratio
        y2 = (y2 - pad_top) / ratio

        # 限制到原始DICOM像素边界。
        x1 = max(
            0.0,
            min(
                float(x1),
                original_width - 1.0,
            ),
        )
        x2 = max(
            0.0,
            min(
                float(x2),
                original_width - 1.0,
            ),
        )
        y1 = max(
            0.0,
            min(
                float(y1),
                original_height - 1.0,
            ),
        )
        y2 = max(
            0.0,
            min(
                float(y2),
                original_height - 1.0,
            ),
        )

        if x2 <= x1 or y2 <= y1:
            continue

        candidates.append(
            {
                "slice_index": slice_index,
                "confidence": float(
                    scores[index]
                ),
                "region_type": "bbox",
                "region": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            }
        )

    return candidates
