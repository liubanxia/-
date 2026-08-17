from model_adapters.report_teacher_runtime import ReportTeacherRuntime


PREFERRED_TEACHERS = [
    "MedGemma-1.5-4B::language_model",
    "14_MAIRA_2_ModelScope::language_model",
    "Hulu-Med-4B::language_model",
    "HealthGPT-Pro-4B::language_model",
]


class ReportTeacherPool:

    def __init__(self):
        self.pool = {
            name: ReportTeacherRuntime(name)
            for name in PREFERRED_TEACHERS
        }

    def select(self, allowed=None):
        allowed = allowed or PREFERRED_TEACHERS

        for name in PREFERRED_TEACHERS:
            if name in allowed and name in self.pool:
                return self.pool[name]

        return None

    def all(self):
        return dict(self.pool)


REPORT_TEACHER_POOL = ReportTeacherPool()
