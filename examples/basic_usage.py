from phoenix_core import analyze_folder

# Use only de-identified or synthetic DICOM data in public examples.
result = analyze_folder("path/to/deidentified_dicom_folder", model_root="path/to/local/models")
print(result["execution_summary"])
print(result["analysis"].report_draft)
