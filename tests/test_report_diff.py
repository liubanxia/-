from report_learning.report_diff_engine import ReportDiffEngine


def test_report_diff_detects_replacement():
    engine = ReportDiffEngine()
    result = engine.compare("影像所见：\n双肺清晰。\n诊断意见：\n未见异常。", "影像所见：\n右肺见结节。\n诊断意见：\n右肺结节。")
    assert result["exact_match"] is False
    assert result["summary"]["change_count"] >= 1
