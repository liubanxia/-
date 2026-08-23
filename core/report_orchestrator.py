from core.report_teacher_pool import REPORT_TEACHER_POOL
from core.radiology_report_draft import RADIOLOGY_REPORT_DRAFT


class PhoenixReportOrchestrator:
    def build_prompt(self, case_info, fused_result):
        draft = RADIOLOGY_REPORT_DRAFT.build(case_info, fused_result)
        return f"""你是放射科影像报告整理模型。

请根据下面已经提取的影像征象整理成规范报告。

要求：
1. 不增加输入中不存在的病灶。
2. 不把影像表现写成病理或实验室确诊。
3. 保持部位、左右侧、数量和范围一致。
4. 使用专业、简洁的放射科语言。
5. 不输出模型名、置信度、评分或分析过程。
6. 只输出“影像所见”和“诊断意见”两部分。
7. 如果证据不足，使用“考虑”“倾向”“建议结合临床”等影像学措辞。

原始结构化草稿：

{draft}
"""

    def generate(self, case_info, fused_result, allowed_teachers=None):
        teacher = REPORT_TEACHER_POOL.select(allowed_teachers)
        if teacher is None:
            return RADIOLOGY_REPORT_DRAFT.build(case_info, fused_result)
        prompt = self.build_prompt(case_info, fused_result)
        teacher.load("cpu")
        try:
            return teacher.generate(prompt, max_new_tokens=512)
        finally:
            teacher.unload()


REPORT_ORCHESTRATOR = PhoenixReportOrchestrator()
