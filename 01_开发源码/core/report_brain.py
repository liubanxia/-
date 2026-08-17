class PhoenixReportBrain:
    """
    把视觉专家结果转换为报告生成上下文。
    不在这里自动调用大模型。
    """

    def build_context(
        self,
        case_info,
        fused_result,
    ):
        findings = []

        for item in fused_result.findings:
            entry = {
                "location": item.location,
                "finding": item.finding,
                "impression": item.impression,
            }

            # 临床输出不暴露模型名和score
            findings.append(entry)

        return {
            "case": {
                "modality": case_info.get(
                    "modality"
                ),
                "body_part": case_info.get(
                    "body_part"
                ),
                "study_description":
                    case_info.get(
                        "study_description"
                    ),
            },

            "findings": findings,

            "instructions": {
                "style": "radiology_structured_report",
                "avoid_pathology_claim": True,
                "doctor_review_required": True,
                "include_model_scores": False,
                "include_model_names": False,
            },
        }

    def teacher_candidates(
        self,
        execution_plan,
    ):
        return list(
            execution_plan.get(
                "report_teachers",
                []
            )
        )


REPORT_BRAIN = PhoenixReportBrain()
