# Synthetic Case Example

This directory contains reproducible synthetic DICOM utilities for Project Phoenix Core. The generated files contain no real patient data and are intended only for development, CI, and routing demonstrations.

## Principles

- Never use identifiable patient data.
- Keep clinical datasets outside the repository.
- Use the bundled generator for public tests and examples.
- Provide local model paths explicitly when testing real adapters.
- Validate model outputs with human review.

## Generate tiny synthetic CT and DR files

```python
from pathlib import Path

from examples.synthetic_case.generate import (
    create_synthetic_ct_series,
    create_synthetic_dr_image,
)

root = Path("synthetic_output")
ct_files = create_synthetic_ct_series(root / "head_ct", body_part="HEAD", count=4)
dr_file = create_synthetic_dr_image(root / "chest_dr.dcm", body_part="CHEST")

print(ct_files)
print(dr_file)
```

The generated datasets use explicit synthetic identifiers such as `SYNTHETIC^PHOENIX` and `SYNTHETIC-NO-PHI`. They are deliberately tiny so they can be generated during automated tests without storing binary DICOM fixtures in Git.

## Example public workflow

1. Generate a synthetic CT or DR case.
2. Build a lightweight case object or load the files through a public DICOM adapter.
3. Exercise routing and result-fusion logic.
4. Run local model adapters only when their weights are available outside the repository.
5. Inspect structured outputs and report drafts with clinician review.

No hospital-specific PACS configuration, credentials, private model weights, or identifiable clinical data belong in this example.
