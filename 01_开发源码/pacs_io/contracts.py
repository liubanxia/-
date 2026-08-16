from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DicomSeries:
    study_uid: str = ""
    series_uid: str = ""
    modality: str = ""
    files: List[Path] = field(default_factory=list)


@dataclass
class CaseInput:
    case_id: str
    series: List[DicomSeries] = field(default_factory=list)
    source: str = "local"
    temp_dir: Optional[Path] = None
