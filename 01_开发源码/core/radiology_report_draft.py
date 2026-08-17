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

        body_part = str(case_info.get("body_part") or "")
        modality = str(case_info.get("modality") or "")

        lines = [
            f"检查类型：{modality}",
            f"检查部位：{body_part}",
            "",
            "影像所见：",
        ]

        for i, text in enumerate(findings, 1):
            lines.append(f"{i}. {text}")

        lines += [
            "",
            "诊断意见：",
        ]

        for i, text in enumerate(impressions, 1):
            lines.append(f"{i}. {text}")

        return "\n".join(lines)


RADIOLOGY_REPORT_DRAFT = RadiologyReportDraftBuilder()
