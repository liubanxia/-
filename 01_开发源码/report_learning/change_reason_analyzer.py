import re


class ChangeReasonAnalyzer:
    """
    医生报告修改类型分析。

    只在RAM中处理Diff。
    不写文件，不保存病例正文。
    """

    def classify(self, change):
        ai = str(
            change.get("ai_text", "")
        )

        doctor = str(
            change.get("final_text", "")
        )

        section = str(
            change.get("section", "")
        )

        result = {
            "section": section,
            "change_type":
                change.get("change_type"),
            "category": "文字调整",
            "possible_reason": "表述修订",
        }

        signals = []

        # 1. 测量值变化
        pattern = (
            r"\d+(?:\.\d+)?\s*"
            r"(?:mm|cm|ml|mL|HU|度|°)"
        )

        ai_values = re.findall(
            pattern,
            ai,
            flags=re.I,
        )

        doctor_values = re.findall(
            pattern,
            doctor,
            flags=re.I,
        )

        if (
            ai_values
            and doctor_values
            and ai != doctor
        ):
            signals.append(
                "测量值修正"
            )

        # 2. 左右侧变化
        ai_side = re.findall(
            r"左侧|右侧|左|右|双侧",
            ai,
        )

        doctor_side = re.findall(
            r"左侧|右侧|左|右|双侧",
            doctor,
        )

        if (
            ai_side
            and doctor_side
            and ai_side != doctor_side
        ):
            signals.append(
                "左右侧修正"
            )

        # 3. 病灶新增 / 删除
        if (
            section == "检查所见"
            and change.get("change_type") == "add"
        ):
            signals.append(
                "病灶或征象新增"
            )

        if (
            section == "检查所见"
            and change.get("change_type") == "delete"
        ):
            signals.append(
                "病灶或征象删除"
            )

        # 4. 诊断结论改变
        if (
            section == "诊断意见"
            and ai != doctor
        ):
            signals.append(
                "诊断结论改变"
            )

        # 5. 诊断确定度变化
        certainty_terms = [
            "考虑",
            "可能",
            "倾向",
            "疑似",
            "符合",
            "明确",
            "不能除外",
        ]

        ai_certainty = [
            x for x in certainty_terms
            if x in ai
        ]

        doctor_certainty = [
            x for x in certainty_terms
            if x in doctor
        ]

        if ai_certainty != doctor_certainty:
            if ai_certainty or doctor_certainty:
                signals.append(
                    "诊断确定度改变"
                )

        # 6. 随访 / 检查建议变化
        advice_terms = [
            "建议",
            "随访",
            "复查",
            "进一步检查",
            "CT",
            "MRI",
            "MR",
            "增强",
            "结合临床",
            "结合既往",
        ]

        ai_advice = [
            x for x in advice_terms
            if x in ai
        ]

        doctor_advice = [
            x for x in advice_terms
            if x in doctor
        ]

        if ai_advice != doctor_advice:
            if ai_advice or doctor_advice:
                signals.append(
                    "随访或检查建议改变"
                )

        if signals:
            result["category"] = " / ".join(
                signals
            )

            result["possible_reason"] = (
                "医生复核影像后进行了内容修正；"
                "具体原因需医生确认"
            )

        result["signals"] = signals

        return result
