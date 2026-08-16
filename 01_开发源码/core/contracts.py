from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Lesion:
    label: str
    confidence: float = 0.0
    series_uid: str = ""
    image_index: Optional[int] = None
    point: Optional[Tuple[int, int]] = None


@dataclass
class AnalysisResult:
    diagnosis: List[str] = field(default_factory=list)
    lesions: List[Lesion] = field(default_factory=list)
    report_draft: str = ""
