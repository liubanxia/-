from dataclasses import dataclass, field
from typing import List


@dataclass
class StructuredReport:
    findings: List[str] = field(default_factory=list)
    impression: List[str] = field(default_factory=list)
    note: str = ""

    def render(self):
        findings = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(self.findings)) or "未形成可靠影像学所见。"
        impression = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(self.impression)) or "未形成可靠影像学诊断意见。"
        text = "影像所见：\n" f"{findings}\n\n" "诊断意见：\n" f"{impression}"
        if self.note:
            text += f"\n\n备注：{self.note}"
        return text
