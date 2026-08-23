from .structured_report import StructuredReport


def _incomplete_report(result, message):
    report = StructuredReport(findings=[message], impression=["当前AI结果不足以形成完整影像学诊断意见，请由影像科医师结合原始影像完成阅片。"])
    result.diagnosis = []; result.report_draft = report.render(); return result


def generate_report(result, incomplete_models=None):
    if incomplete_models:
        return _incomplete_report(result, "本次AI分析未完整完成，关键分析模块未成功执行。")
    diagnostic_selected = result.execution_summary.get("diagnostic_models_selected", [])
    if not diagnostic_selected:
        return _incomplete_report(result, "当前仅完成解剖路由或辅助处理，尚无适用的疾病诊断模型完成分析。")
    if not result.diagnostic_executed:
        return _incomplete_report(result, "疾病诊断模型尚未实际执行完成，当前结果不能用于判断有无病灶。")
    if result.lesions:
        counts = {}
        for lesion in result.lesions: counts[lesion.label] = counts.get(lesion.label, 0) + 1
        report = StructuredReport(
            findings=[f"检出{label}相关影像异常，共{count}处候选区域，具体部位、范围及形态请结合病灶定位图复核。" for label, count in counts.items()],
            impression=[f"影像学考虑{label}相关改变，需医师复核。" for label in counts],
        )
        result.diagnosis = list(counts); result.report_draft = report.render(); return result
    result.diagnosis = []
    result.report_draft = StructuredReport(findings=["已选择的疾病诊断模型本次未输出候选病灶。"], impression=["该结果不等同于影像学阴性诊断；请结合原始影像完成最终诊断。"]).render()
    return result
