CHEST_NAMES = {
    "Lung Opacity": "肺部阴影",
    "Atelectasis": "肺不张",
    "Effusion": "胸腔积液",
    "Pneumonia": "肺炎/实变候选",
    "Cardiomegaly": "心影增大",
    "Edema": "肺水肿",
    "Pneumothorax": "气胸",
    "Consolidation": "肺实变",
    "Mass": "肺部肿块",
    "Nodule": "肺结节",
    "Pleural Thickening": "胸膜增厚",
    "Fibrosis": "肺纤维化",
    "Emphysema": "肺气肿",
    "Infiltration": "肺浸润影",
    "Hernia": "疝",
}


def _get_chest_candidates(result):
    raw = result.raw_model_results.get(
        "torchxrayvision_chest",
        {},
    )

    items = raw.get(
        "ranked_candidates",
        [],
    )

    return items[:5]


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

    chest = _get_chest_candidates(result)

    if chest:
        diagnosis = []
        findings = []

        for item in chest:
            label = item["label"]
            score = float(item["score"])
            name = CHEST_NAMES.get(label, label)

            diagnosis.append(
                f"{name}（候选评分 {score:.3f}）"
            )

            findings.append(
                f"{name} {score:.3f}"
            )

        result.diagnosis = diagnosis

        result.report_draft = (
            "影像所见：胸片AI多病种候选评分："
            + "；".join(findings)
            + "。\n"
            "诊断意见：以上为模型候选评分，"
            "尚未经过本系统临床阈值校准，"
            "不等同于疾病阳性诊断，"
            "请由影像科医师结合影像复核。"
        )

        return result

    if not result.lesions:
        result.diagnosis = [
            "当前已运行模型未检出明确目标病灶"
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

    counts = {}

    for lesion in result.lesions:
        counts[lesion.label] = (
            counts.get(lesion.label, 0) + 1
        )

    findings = "；".join(
        f"{label}（{count}处候选区域）"
        for label, count in counts.items()
    )

    result.report_draft = (
        f"影像所见：AI检出：{findings}。\n"
        f"诊断意见：{'；'.join(labels)}。"
    )

    return result
