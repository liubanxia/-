from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass
class TranslationLearningRecord:
    source: str
    model1_text: str
    model2_text: str
    final_text: str
    final_backend: str
    domain: str = "medical"
    reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["created_at"] = datetime.now(timezone.utc).isoformat()
        return value


class TranslationLearningCollector:
    """Store approved translation corrections for future Phoenix model tuning.

    This stores only Phoenix-owned learning data. It does not update model
    weights online; it creates a clean dataset for later supervised tuning.
    """

    def __init__(self, root: str | Path):
        self.path = Path(root) / "translation_learning.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TranslationLearningRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def export_training_pair(self, output: str | Path) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        output.write_text(
            "\n".join(
                json.dumps(
                    {
                        "instruction": "医学专业翻译，请保持术语、数字、单位和否定关系准确。",
                        "input": row["source"],
                        "output": row["final_text"],
                    },
                    ensure_ascii=False,
                )
                for row in rows
            ),
            encoding="utf-8",
        )
