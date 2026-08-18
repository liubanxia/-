from .structured_report import StructuredReport


CHEST_NAMES = {
    "Lung Opacity": "肺部异常密度影",
    "Atelectasis": "肺不张",
    "Effusion": "胸腔积液",
    "Pneumonia": "炎症/实变改变",
    "Cardiomegaly": "心影增大",
    "Edema": "肺水肿",
    "Pneumothorax": "气胸",
    "Consolidation": "肺实变",
    "Mass": "肺部肿块",
    "Nodule": "肺结节",
    "Pleural Thickening": "胸膜增厚",
    "Fibrosis": "肺纤维化改变",
    "Emphysema": "肺气肿改变",
    "Infiltration": "肺浸润性改变",
    "Hernia": "疝",
}


def _chest_screening_labels(result):
    raw = result.raw_model_results.get(
        "torchxrayvision_chest",
        {},
    )

    items = raw.get("ranked_candidates", [])

    return [
        CHEST_NAMES.get(
            item.get("label", ""),
            item.get("label", ""),
        )
        for item in items
        if item.get("label")
    ]


def _incomplete_report(result, message):
    report = StructuredReport(
        findings=[message],
        impression=[
            "当前AI结果不足以形成完整影像学诊断意见，请由影像科医师结合原始影像完成阅片。"
        ],
    )

    result.diagnosis = []
    result.report_draft = report.render()
    return result


def generate_report(result, incomplete_models=None):
    incomplete_models = incomplete_models or []

    if incomplete_models:
        return _incomplete_report(
            result,
            "本次AI分析未完整完成，部分分析模块未成功执行。",
        )

    diagnostic_models_selected = result.execution_summary.get(
        "diagnostic_models_selected",
        [],
    )

    # Critical safety rule: a router/segmentation model executing successfully
    # is not equivalent to a disease-diagnostic model having run.
    if not diagnostic_models_selected:
        return _incomplete_report(
            result,
            "当前仅完成解剖路由或辅助处理，尚无适用的疾病诊断模型完成分析。",
        )

    if not result.diagnostic_executed:
        return _incomplete_report(
            result,
            "疾病诊断模型尚未实际执行完成，当前结果不能用于判断有无病灶。",
        )

    if result.lesions:
        counts = {}

        for lesion in result.lesions:
            counts[lesion.label] = (
                counts.get(lesion.label, 0) + 1
            )

        findings = [
            (
                f"检出{label}相关影像异常，"
                f"共{count}处候选区域，"
                "具体部位、范围及形态请结合病灶定位图复核。"
            )
            for label, count in counts.items()
        ]

        impression = [
            f"影像学考虑{label}相关改变，需医师复核。"
            for label in counts
        ]

        report = StructuredReport(
            findings=findings,
            impression=impression,
        )

        result.diagnosis = list(counts)
        result.report_draft = report.render()
        return result

    chest = _chest_screening_labels(result)

    if chest:
        report = StructuredReport(
            findings=[
                "胸片AI筛查检测到异常影像信号，当前分类模型不能可靠提供病灶部位、形态及范围。"
            ],
            impression=[
                "需结合原始胸片及后续通用视觉/定位模型完成影像学诊断。"
            ],
        )

        result.diagnosis = []
        result.report_draft = report.render()
        return result

    report = StructuredReport(
        findings=[
            "已选择的疾病诊断模型本次未输出候选病灶。"
        ],
        impression=[
            "该结果仅表示本次AI模型未输出候选异常，不等同于影像学阴性诊断；请结合原始影像完成最终诊断。"
        ],
    )

    result.diagnosis = []
    result.report_draft = report.render()
    return result
