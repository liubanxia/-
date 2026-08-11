"""
Project Phoenix 配置文件
"""

from pathlib import Path

# 项目名称
PROJECT_NAME = "Project Phoenix"

# 当前版本
VERSION = "V1.0.0"

# 开发模式
DEBUG = True

# 支持影像类型
SUPPORTED_MODALITIES = [
    "CT",
    "DX",
]

MODALITY_DISPLAY_NAMES = {
    "CT": "CT",
    "DX": "DR",
}

# --------------------------------------------------
# AI 模型路径
# --------------------------------------------------

# Project Phoenix 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# AI 模型总目录
AI_MODEL_ROOT = PROJECT_ROOT / "04_AI模型"

# 视觉A：常规综合阅片AI
VISUAL_A_MODEL_DIR = AI_MODEL_ROOT / "视觉A_综合阅片"

# 视觉B：骨折漏诊防护AI
VISUAL_B_MODEL_DIR = AI_MODEL_ROOT / "视觉B_骨折防护"

# 当前尚未正式选定医学模型。
# 文件名必须在模型正式确定后显式填写；
# 禁止程序自动猜测、自动搜索或自动选择其他模型。
VISUAL_A_MODEL_FILENAME = ""
VISUAL_B_MODEL_FILENAME = ""
