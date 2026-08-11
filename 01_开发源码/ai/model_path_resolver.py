from pathlib import Path

from config.settings import (
    VISUAL_A_MODEL_DIR,
    VISUAL_A_MODEL_FILENAME,
    VISUAL_B_MODEL_DIR,
    VISUAL_B_MODEL_FILENAME,
)


def _resolve_model_path(model_dir, model_filename, visual_name):
    """
    严格解析Phoenix视觉AI模型路径。

    安全原则：
    - 模型文件名必须显式配置；
    - 不允许自动搜索目录；
    - 不允许猜测其他模型；
    - 不允许通过文件名跳出指定模型目录；
    - 当前只允许.onnx模型；
    - 文件不存在时安全停止。
    """

    model_dir = Path(model_dir)
    filename = str(model_filename).strip()

    if not filename:
        raise RuntimeError(
            f"{visual_name}模型文件名尚未配置"
        )

    filename_path = Path(filename)

    if filename_path.name != filename:
        raise ValueError(
            f"{visual_name}模型文件名不得包含目录路径：{filename}"
        )

    if filename_path.suffix.lower() != ".onnx":
        raise ValueError(
            f"{visual_name}当前只允许.onnx模型：{filename}"
        )

    model_path = model_dir / filename

    if not model_dir.exists():
        raise FileNotFoundError(
            f"{visual_name}模型目录不存在：{model_dir}"
        )

    if not model_dir.is_dir():
        raise ValueError(
            f"{visual_name}模型目录异常：{model_dir}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"{visual_name}模型文件不存在：{model_path}"
        )

    if not model_path.is_file():
        raise ValueError(
            f"{visual_name}模型路径不是文件：{model_path}"
        )

    return model_path


def resolve_visual_a_model_path():
    """解析视觉A：综合阅片AI模型路径。"""

    return _resolve_model_path(
        VISUAL_A_MODEL_DIR,
        VISUAL_A_MODEL_FILENAME,
        "视觉A",
    )


def resolve_visual_b_model_path():
    """解析视觉B：骨折漏诊防护AI模型路径。"""

    return _resolve_model_path(
        VISUAL_B_MODEL_DIR,
        VISUAL_B_MODEL_FILENAME,
        "视觉B",
    )
