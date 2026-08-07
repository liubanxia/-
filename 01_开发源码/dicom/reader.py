from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

from config.settings import SUPPORTED_MODALITIES
from utils.logger import setup_logger


logger = setup_logger()


def read_dicom(file_path):
    """
    安全读取单个 DICOM 文件。

    当前版本：
    - 仅支持配置允许的影像类型
    - 不做医学诊断
    - 不修改原始 DICOM 文件
    """

    path = Path(file_path)

    if not path.exists():
        logger.error("DICOM 文件不存在：%s", path)
        raise FileNotFoundError(f"DICOM 文件不存在：{path}")

    if not path.is_file():
        logger.error("目标不是文件：%s", path)
        raise ValueError(f"目标不是文件：{path}")

    # 只把真正的 DICOM 读取过程放在 try 内
    try:
        dataset = pydicom.dcmread(path)

    except InvalidDicomError as exc:
        logger.error("无效 DICOM 文件：%s", path)
        raise ValueError(
            "文件不是有效的 DICOM 文件"
        ) from exc

    except Exception:
        logger.exception(
            "读取 DICOM 时发生未知错误：%s",
            path,
        )
        raise

    # Modality 检查放在 try / except 外面，
    # 避免程序主动拒绝 MR 时被当成未知异常。
    modality = getattr(dataset, "Modality", "UNKNOWN")

    if modality not in SUPPORTED_MODALITIES:
        logger.warning(
            "当前版本不支持该影像类型 | Modality=%s | File=%s",
            modality,
            path.name,
        )
        raise ValueError(
            f"当前版本不支持影像类型：{modality}"
        )

    logger.info(
        "DICOM 读取成功 | Modality=%s | File=%s",
        modality,
        path.name,
    )

    return dataset