# Changelog

All notable changes to Project Phoenix Core are documented here.

## Unreleased

### Added
- Reproducible synthetic CT and DR DICOM generator for public development and CI
- Automated routing tests covering synthetic head CT, abdomen CT router output, chest DR, and bone DR
- Result-fusion regression tests for lesion geometry, confidence normalization, metadata preservation, and malformed payload handling

### Security
- Synthetic fixtures are generated at test time and contain explicit non-PHI identifiers
- No hospital PACS configuration, credentials, patient data, or private model weights are required by the new tests

## v0.1.0 - 2026-08-23

### Added
- Public open-source core repository
- DICOM-first medical imaging pipeline structure
- CT and DR workflow foundations
- Model adapter architecture
- Result fusion interfaces
- Structured report drafting interfaces
- Synthetic/de-identified example documentation

### Security
- No patient data included
- No hospital credentials included
- No private deployment configuration included
- No private model weights included
