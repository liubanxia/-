from __future__ import annotations

"""Blank-student production integration without per-document wrapper stacking."""

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import translation_blank_student as student
    from . import translation_survival_memory as survival

    old_store = survival._store_decision

    def store_decision(engine, source: str, target: str, decision) -> None:
        old_store(engine, source, target, decision)
        student._observe_decision(engine, source, target, decision)

    store_decision._phoenix_blank_student_observer_v2 = True
    survival._store_decision = store_decision

    _INSTALLED = True
    params = student.parameter_count("seed")
    print(
        "[Phoenix][空白学生] v2已启用：随机初始化Byte-GRU从第1份资料开始收集/影子学习；"
        f"seed参数={params:,}，原始FP32权重约{params * 4 / 1024 / 1024:.2f}MB。"
        "文档结束训练由统一后处理器触发；永不参与正式译文。",
        flush=True,
    )
