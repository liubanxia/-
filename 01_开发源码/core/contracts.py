from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Lesion:
    label: str
    confidence: float = 0.0
    series_uid: str = ""
    image_index: Optional[int] = None
    point: Optional[Tuple[float, float]] = None
    box: Optional[Tuple[float, ...]] = None
    box_3d: Optional[Tuple[float, ...]] = None
    world_point_lps: Optional[Tuple[float, float, float]] = None
    geometry_mode: str = ""
    voxel_count: int = 0
    source_model: str = ""
    finding: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    diagnosis: List[str] = field(default_factory=list)
    lesions: List[Lesion] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report_draft: str = ""
    raw_model_results: dict = field(default_factory=dict)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    diagnostic_executed: bool = False
    diagnostic_valid: bool = False
