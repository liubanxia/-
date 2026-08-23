from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ReviewStageResult:
    stage: str
    text: str
    changed: bool
    passed: bool
    reason: str = ""


class MedicalTranslationReviewPipeline:
    """Second-pass review chain.

    Translation and review are separated:
    model1/2/3/API create translation first;
    this pipeline performs staged review and correction afterwards.
    """

    def __init__(self, reviewers: dict[str, Callable[[str], str]]):
        self.reviewers = reviewers

    def run(self, text: str) -> tuple[str, tuple[ReviewStageResult, ...]]:
        current = str(text or "")
        results = []

        for stage in ("model1_review", "model2_review", "model3_review", "api_review"):
            reviewer = self.reviewers.get(stage)
            if reviewer is None:
                continue
            updated = str(reviewer(current) or current)
            results.append(
                ReviewStageResult(
                    stage=stage,
                    text=updated,
                    changed=updated != current,
                    passed=True,
                )
            )
            current = updated

        return current, tuple(results)
