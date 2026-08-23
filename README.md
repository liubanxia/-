# Project Phoenix Core

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![DICOM](https://img.shields.io/badge/Medical%20Imaging-DICOM-green.svg)](https://www.dicomstandard.org/)

**Offline, DICOM-first medical-imaging AI framework for CT, DR/X-ray and extensible specialist pipelines.**

Project Phoenix Core is the public engineering core extracted from a larger offline radiology-assistance project. It focuses on portable image ingestion, explicit model routing, lightweight inference adapters, result fusion, lesion geometry, and clinician-reviewed report drafting.

> Research and engineering software only. It is not a medical device and does not replace clinician interpretation.

## Overview

Phoenix Core explores how medical imaging AI components can be composed into transparent, modular workflows while keeping the human clinician in control.

Core ideas:

- DICOM-first processing instead of screenshot-based analysis
- Explicit routing between imaging tasks and specialist models
- Modular adapters for different AI backends
- Structured intermediate results for explainability
- Human-reviewed reporting workflows
- Offline-capable engineering design

## Documentation

- [Architecture](docs/architecture.md)
- [Demo Flow](docs/demo.md)
- [Synthetic Example](examples/synthetic_case/README.md)
- [Roadmap](ROADMAP.md)

## What is included

- DICOM folder ingestion with Study/Series isolation and spatial slice sorting
- Local Orthanc adapter
- CT two-stage routing with BodyPartRegression integration
- DR/X-ray routing and YOLO detection/segmentation adapters
- MedSAM2, SAM-Med3D, TotalSegmentator, VISTA3D and SegVol adapter interfaces
- Local Hugging Face encoder/report-teacher adapters
- CPU-only ONNX adapter with lazy loading
- Doctor-triggered inference gate
- Unified lesion/result contracts and world-LPS → DICOM pixel mapping
- Structured radiology report drafting and report-diff utilities
- Hardware-aware model loading and explicit incomplete-analysis states
- Case-scoped temporary storage and cleanup

## What is deliberately not included

- Vendor-specific hospital/PACS integrations or write-back automation
- Real patient images, reports, identifiers or clinical databases
- Credentials, API keys, certificates or private network settings
- Workstation/SSD-specific paths
- Model weights or generated clinical outputs
- Private repository Git history

See [PUBLIC_EXTRACTION.md](PUBLIC_EXTRACTION.md) for the public/private boundary.

## Quick start

```bash
python -m pip install -r requirements.txt
```

```python
from phoenix_core import analyze_folder

result = analyze_folder(
    "path/to/deidentified_dicom_folder",
    model_root="path/to/local/models",
)

print(result["execution_summary"])
print(result["analysis"].report_draft)
```

The base dependency set is intentionally small. Model-specific packages are listed in `requirements-optional.txt` and model weights are supplied separately by the user.

## Design principles

1. **Offline-first** — core processing does not require cloud access.
2. **DICOM-first** — routing and spatial mapping use DICOM metadata/geometry rather than screenshots.
3. **Doctor-controlled** — inference requires explicit activation by the caller.
4. **No false negatives from failures** — missing/failed models remain explicit incomplete states.
5. **Lightweight frontline, large teachers offline** — larger models can support research and distillation workflows.
6. **Ephemeral case data** — temporary case state is removed when the case closes.
7. **Explainable composition** — routing, model execution, fusion and reporting remain inspectable layers.

## Repository layout

```text
ai/                 visual-AI interfaces and safety contracts
ai_models/          generic component registries and CT routing backend
core/               runtime, routing, pipeline, geometry, fusion, reporting
model_adapters/     model-specific adapters
pacs_io/            generic DICOM/Orthanc input adapters
report_learning/    report comparison utilities
output/             generic overlay/result helpers
docs/               architecture documentation
examples/           de-identified/synthetic usage examples
tests/              core behavior tests
```

## Safety and privacy

Never commit identifiable DICOM/clinical data, deployment credentials, private network configuration, or model weights. Contributors remain responsible for reviewing changes before publication.

## Status

Active public-core extraction from Project Phoenix. Interfaces are being stabilized and more synthetic tests/documentation will be added as reusable components are separated from private deployment code.
