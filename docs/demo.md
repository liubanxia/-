# Project Phoenix Core Demo Flow

This document describes the public research workflow.

```text
DICOM Folder / Orthanc Input
            |
            v
     Case Input Adapter
            |
            v
 Doctor-Controlled Inference Gate
            |
            v
       Modality Router
       /            \
     CT              DR/X-ray
     |                 |
 Body Part        Detection /
 Routing           Segmentation
     \                 /
      \               /
          Model Hub
             |
             v
      Result Fusion Layer
             |
             v
 Lesion Geometry + Spatial Mapping
             |
             v
 Structured Report Draft
             |
             v
 Clinician Review
```

## Design goals

- Keep DICOM metadata and spatial geometry as the source of truth.
- Keep model execution modular through adapters.
- Keep inference explicitly controlled by the caller.
- Keep outputs inspectable through structured result objects.

## Public demo requirements

The public repository uses only synthetic or properly de-identified examples.

Users provide their own:

- DICOM data
- model weights
- optional model-specific dependencies

No patient data or deployment credentials are included.
