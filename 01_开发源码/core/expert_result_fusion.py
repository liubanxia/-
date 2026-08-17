from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpertFinding:
    expert_id: str
    task: str
    location: str = ""
    finding: str = ""
    impression: str = ""
    geometry: Any = None

    # 分数仅后台保存，不进入医生报告
    score: float | None = None

    metadata: dict = field(
        default_factory=dict
    )


@dataclass
class UnifiedExpertResult:
    modality: str
    findings: list[ExpertFinding] = field(
        default_factory=list
    )
    backstage: dict = field(
        default_factory=dict
    )


class ExpertResultFusion:

    def fuse(self, modality, results):
        fused = UnifiedExpertResult(
            modality=modality
        )

        seen = set()

        for result in results:
            if result is None:
                continue

            if isinstance(result, ExpertFinding):
                candidates = [result]
            elif isinstance(result, list):
                candidates = [
                    x for x in result
                    if isinstance(x, ExpertFinding)
                ]
            else:
                continue

            for item in candidates:
                key = (
                    item.location.strip().lower(),
                    item.finding.strip().lower(),
                    item.impression.strip().lower(),
                )

                if key in seen:
                    continue

                seen.add(key)
                fused.findings.append(item)

        fused.backstage["expert_count"] = len(
            {
                x.expert_id
                for x in fused.findings
            }
        )

        fused.backstage["finding_count"] = len(
            fused.findings
        )

        return fused


EXPERT_RESULT_FUSION = ExpertResultFusion()
