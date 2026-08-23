# Synthetic Case Example

This directory contains documentation for running Project Phoenix Core with synthetic or properly de-identified DICOM data.

## Principles

- Never use identifiable patient data.
- Keep clinical datasets outside the repository.
- Provide local model paths explicitly.
- Validate outputs with human review.

Example workflow:

1. Prepare a synthetic DICOM folder.
2. Configure local model adapters.
3. Run the analysis pipeline.
4. Inspect structured results and generated report drafts.
