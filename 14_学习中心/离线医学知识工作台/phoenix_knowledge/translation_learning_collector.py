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
    """Collect Phoenix-owned translation corrections.

    Records are training candidates only. They never modify model weights
    online. Only reviewed final translations should enter future tuning sets.
    """

    def __init__(self, root: str | Path):
        self.path = Path(root) / "translation_learning.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TranslationLearningRecord) -> None:
        if not record.source.strip() or not record.final_text.strip():
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def export_training_pair(self, output: str | Path) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pairs = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not bool(row.get("reviewed", False)):
                    continue
                pairs.append(
                    {
                        "instruction": "医学专业翻译，请保持术语、数字、单位、否定关系和解剖关系准确。",
                        "input": row["source"],
                        "output": row["final_text"],
                    }
                )
        output.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in pairs),
            encoding="utf-8",
        )
