from __future__ import annotations

import unittest

from phoenix_knowledge.translation_models import TranslationValidator


class TranslationMedicalSemanticSafetyTests(unittest.TestCase):
    def setUp(self):
        self.validator = TranslationValidator()

    def rejected(self, source: str, translated: str, marker: str):
        report = self.validator.validate(source, translated, "中文")
        self.assertFalse(report.ok, report)
        self.assertTrue(any(marker in x for x in report.reasons), report.reasons)

    def accepted(self, source: str, translated: str):
        report = self.validator.validate(source, translated, "中文")
        self.assertTrue(report.ok, report)

    def test_negation_and_cannot_exclude(self):
        self.rejected("CT demonstrates no pleural effusion and no pneumothorax.", "CT显示胸腔积液及气胸。", "否定关系")
        self.accepted("CT demonstrates no pleural effusion and no pneumothorax.", "CT未见胸腔积液，亦未见气胸。")
        src = "Although indeterminate, malignancy cannot be excluded on the current CT examination."
        self.rejected(src, "虽然性质未定，但当前CT可以排除恶性。", "不能排除")
        self.accepted(src, "虽然性质未定，但当前CT仍不能排除恶性。")

    def test_uncertainty_and_laterality(self):
        src = "The imaging appearance is suggestive of malignancy, but the finding is not diagnostic."
        self.rejected(src, "影像表现明确诊断为恶性。", "诊断确定性")
        self.accepted(src, "影像表现提示恶性可能，但不能据此明确诊断。")
        src = "A 12 mm spiculated pulmonary nodule is present in the right upper lobe."
        self.rejected(src, "左上叶可见一枚12 mm毛刺状肺结节。", "左右侧")
        self.accepted(src, "右上叶可见一枚12 mm毛刺状肺结节。")

    def test_directionality(self):
        self.rejected("The lesion demonstrates increased T2 signal compared with muscle.", "与肌肉相比病灶T2信号减低。", "信号高低")
        self.accepted("The lesion demonstrates increased T2 signal compared with muscle.", "与肌肉相比病灶T2信号增高。")
        self.rejected("The pulmonary nodule increased in size on follow-up.", "随访肺结节较前缩小。", "大小变化")
        self.accepted("The pulmonary nodule increased in size on follow-up.", "随访肺结节较前增大。")
        self.rejected("Arterial phase enhancement was higher than delayed phase enhancement.", "动脉期强化低于延迟期强化。", "比较方向")
        self.accepted("Arterial phase enhancement was higher than delayed phase enhancement.", "动脉期强化高于延迟期强化。")

    def test_polarity(self):
        self.rejected("The lesion is benign on imaging follow-up.", "影像随访显示病灶为恶性。", "良恶性")
        self.accepted("The lesion is benign on imaging follow-up.", "影像随访显示病灶为良性。")
        self.rejected("The patient has acute pancreatitis.", "患者为慢性胰腺炎。", "急慢性")
        self.accepted("The patient has acute pancreatitis.", "患者为急性胰腺炎。")
        self.rejected("A 65-year-old male patient presented with cough.", "一名65岁女性患者因咳嗽就诊。", "性别")
        self.accepted("A 65-year-old male patient presented with cough.", "一名65岁男性患者因咳嗽就诊。")
        self.rejected("The test was negative for malignancy.", "恶性检测结果为阳性。", "阳性/阴性")
        self.accepted("The test was negative for malignancy.", "恶性检测结果为阴性。")
        self.rejected("The lesion is nonenhancing on postcontrast imaging.", "增强后病灶明显强化。", "强化/无强化")
        self.accepted("The lesion is nonenhancing on postcontrast imaging.", "增强后病灶无强化。")

    def test_metric_binding(self):
        src = "Sensitivity and specificity were 82% and 95%, respectively."
        self.rejected(src, "敏感度和特异度分别为95%和82%。", "统计指标数值绑定")
        self.accepted(src, "敏感度和特异度分别为82%和95%。")

    def test_false_positive_guards(self):
        self.accepted("If left untreated, the lesion may progress.", "如不治疗，该病灶可能进展。")
        self.accepted("The above findings support pneumonia.", "上述征象支持肺炎。")
        self.accepted("The growth plate remains open.", "骨骺板仍未闭合。")
        self.accepted("The lesion lies below the diaphragm.", "病灶位于膈下。")


if __name__ == "__main__":
    unittest.main()
