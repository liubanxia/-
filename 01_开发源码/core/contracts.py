from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Lesion:
    label: str
    confidence: float = 0.0
    series_uid: str = ""
    image_index: Optional[int] = None
    point: Optional[Tuple[int, int]] = None
    box: Optional[Tuple[int, int, int, int]] = None
    voxel_count: int = 0
    source_model: str = ""


@dataclass
class AnalysisResult:
    diagnosis: List[str] = field(default_factory=list)
    lesions: List[Lesion] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report_draft: str = ""
    raw_model_results: dict = field(default_factory=dict)
