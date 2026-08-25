from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class TranslationCorrectionSample:
    source: str
    model1_text: str
    model2_text: str = ""
    final_text: str = ""
    final_backend: str = ""
    domain: str = "medical"


class TranslationLearningPool:
    """Collect accepted translation corrections for future legal fine-tuning.

    This pool stores Phoenix-owned correction data only. It does not train from
    restricted model weights or hidden model internals.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or "translation_learning_pool")
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "corrections.jsonl"

    def add(self, sample: TranslationCorrectionSample) -> None:
        payload = asdict(sample)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        with self.file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def count(self) -> int:
        if not self.file.exists():
            return 0
        with self.file.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def export_training_pairs(self) -> list[dict[str, str]]:
        if not self.file.exists():
            return []
        rows = []
        for line in self.file.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("source") and item.get("final_text"):
                rows.append({
                    "source": item["source"],
                    "target": item["final_text"],
                })
        return rows
