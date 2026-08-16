def generate_report(result):
    if not result.lesions:
        result.diagnosis = ["未发现已被当前模型确认的明确异常"]
        result.report_draft = (
            "影像所见：当前AI分析未检出明确异常病灶。\n"
            "诊断意见：请结合完整影像由影像科医师复核。"
        )
        return result

    labels = [x.label for x in result.lesions]
    unique = list(dict.fromkeys(labels))

    result.diagnosis = unique

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
        f"诊断意见：{'；'.join(unique)}。"
    )

    return result
