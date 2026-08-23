# Architecture

Project Phoenix Core is a portable, DICOM-first medical-imaging AI framework.

## Data flow

1. **Input adapter** — folder or local Orthanc input produces a `CaseInput` with DICOM series metadata.
2. **Doctor-controlled inference gate** — visual AI is off by default and must be explicitly enabled by the caller/UI.
3. **Routing** — modality and DICOM metadata select the first-stage router and lightweight specialist candidates.
4. **Model hub** — adapters are lazily loaded and execution failures remain explicit rather than being interpreted as negative studies.
5. **Result fusion** — model outputs are normalized into `AnalysisResult` and `Lesion` objects.
6. **Spatial resolution** — world LPS coordinates can be mapped back to DICOM slice/pixel coordinates.
7. **Report layer** — structured findings are converted into a draft requiring clinician review.
8. **Cleanup** — case-scoped temporary data are deleted on case close.

## Public/private boundary

This repository deliberately contains no vendor-specific hospital integration, real patient data, credentials, private network configuration, model weights, or workstation-specific paths. Those concerns are expected to live in external deployment adapters/configuration.
