class RadiologyReportDraftBuilder:
    def build(self, case_info, fused_result):
        findings = []
        impressions = []
        for item in getattr(fused_result, "findings", []):
            location = (item.location or "").strip()
            finding = (item.finding or "").strip()
            impression = (item.impression or "").strip()
            if finding:
                text = f"{location}：{finding}" if location else finding
                if text not in findings:
                    findings.append(text)
            if impression and impression not in impressions:
                impressions.append(impression)
        if not findings:
            findings = ["待AI进一步整理影像征象。"]
        if not impressions:
            impressions = ["结合影像所见进一步判断。"]
        lines = [
            f"检查类型：{str(case_info.get('modality') or '')}",
            f"检查部位：{str(case_info.get('body_part') or '')}",
            "", "影像所见：",
        ]
        lines += [f"{i}. {text}" for i, text in enumerate(findings, 1)]
        lines += ["", "诊断意见："]
        lines += [f"{i}. {text}" for i, text in enumerate(impressions, 1)]
        return "\n".join(lines)


RADIOLOGY_REPORT_DRAFT = RadiologyReportDraftBuilder()
