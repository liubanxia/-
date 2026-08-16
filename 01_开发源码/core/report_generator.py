def generate_report(result, incomplete_models=None):
    incomplete_models = incomplete_models or []

    if incomplete_models:
        names = "、".join(incomplete_models)

        result.diagnosis = [
            "AI分析未完整完成"
        ]

        result.report_draft = (
            f"AI分析未完整完成：{names} 未成功执行。\n"
            "当前结果不能用于判断影像阴性，"
            "请由影像科医师完成阅片。"
        )
        return result

    if not result.lesions:
        result.diagnosis = [
            "当前已运行模型未检出明确病灶"
        ]

        result.report_draft = (
            "影像所见：当前已成功运行的AI模型"
            "未检出明确目标病灶。\n"
            "诊断意见：AI辅助结果，仅供影像科医师复核。"
        )
        return result

    labels = list(
        dict.fromkeys(
            x.label for x in result.lesions
        )
    )

    result.diagnosis = labels

    findings = "；".join(
        f"{x.label}"
        + (
            f"（置信度{x.confidence:.2f}）"
            if x.confidence else ""
        )
        for x in result.lesions
    )

    result.report_draft = (
        f"影像所见：AI检出：{findings}。\n"
        f"诊断意见：{'；'.join(labels)}。"
    )

    return result
