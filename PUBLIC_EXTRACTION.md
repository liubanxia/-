# Public extraction policy

Project Phoenix Core is published as a clean source snapshot rather than a mirror of the private deployment repository.

The public repository must not contain:

- hospital-specific PACS/YUNPACS integrations or deployment configuration;
- patient images, reports, identifiers, databases, or other clinical data;
- credentials, API keys, tokens, certificates, or private network configuration;
- local workstation/SSD paths or environment-specific configuration;
- model weights, generated outputs, caches, logs, or temporary files;
- private repository Git history.

Generic DICOM processing, AI pipeline orchestration, model adapters, inference runtime, result fusion, report-generation logic, tests based on synthetic/non-clinical data, and general documentation may be included.
