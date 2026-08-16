from dataclasses import dataclass


@dataclass
class RunConfig:
    source: str = "folder"
    output_mode: str = "A"
    orthanc_url: str = "http://127.0.0.1:8042"
