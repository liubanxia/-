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

    # 分数只留后台。
    # 当前未做临床阈值校准，因此不能据此直接生成疾病确诊。
    return [
        CHEST_NAMES.get(
            item.get("label", ""),
            item.get("label", ""),
        )
        for item in items
        if item.get("label")
    ]


def generate_report(result, incomplete_models=None):
    incomplete_models = incomplete_models or []

    if incomplete_models:
        report = StructuredReport(
            findings=[
                "本次AI分析未完整完成，部分分析模块未成功执行。"
            ],
            impression=[
                "当前结果不足以形成完整影像学诊断意见，请由影像科医师结合原始影像完成阅片。"
            ],
        )

        result.diagnosis = []
        result.report_draft = report.render()
        return result

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
            f"影像学考虑{label}相关改变。"
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
                "胸片AI筛查检测到异常影像信号，"
                "当前分类模型不能可靠提供病灶部位、形态及范围。"
            ],
            impression=[
                "需结合原始胸片及后续通用视觉/定位模型完成影像学诊断。"
            ],
        )

        # 分类评分仍保留在 raw_model_results，
        # 但不进入医生报告。
        result.diagnosis = []
        result.report_draft = report.render()
        return result

    report = StructuredReport(
        findings=[
            "当前已运行的AI分析模块未形成明确可报告影像异常。"
        ],
        impression=[
            "请结合原始影像完成影像科诊断。"
        ],
    )

    result.diagnosis = []
    result.report_draft = report.render()

    return result
