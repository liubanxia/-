from __future__ import annotations

"""医学资料翻译策略补丁。

历史请求分类兼容层。正式 PDF/PPTX/DOCX 入口会无条件使用 Smart2；
本模块不再有权把正式文档降级到旧 Smart1 翻译链。
"""

MEDICAL_KEYWORDS = {
    "medical", "medicine", "radiology", "radiology", "ct", "mri",
    "pathology", "anatomy", "clinical", "diagnosis", "guideline",
    "影像", "医学", "临床", "诊断", "解剖", "指南",
}


def is_medical_document(name: str) -> bool:
    text = (name or "").lower()
    return any(k.lower() in text for k in MEDICAL_KEYWORDS)


def preferred_translation_level(name: str, requested: str | None = None) -> str:
    """医学资料优先质量模式，普通资料保留快速模式。"""
    if is_medical_document(name):
        return "smart2"
    return requested or "smart1"


def should_use_local_fast_translation(name: str) -> bool:
    return not is_medical_document(name)
