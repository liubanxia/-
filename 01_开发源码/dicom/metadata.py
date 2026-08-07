def extract_metadata(dataset):
    """
    提取非身份性 DICOM 基础元数据。

    当前版本不提取：
    - PatientName
    - PatientID
    - PatientBirthDate
    - 其他患者身份信息
    """

    metadata = {
        "Modality": getattr(dataset, "Modality", None),
        "StudyDate": getattr(dataset, "StudyDate", None),
        "SeriesDescription": getattr(dataset, "SeriesDescription", None),
        "Rows": getattr(dataset, "Rows", None),
        "Columns": getattr(dataset, "Columns", None),
        "PixelSpacing": getattr(dataset, "PixelSpacing", None),
        "SliceThickness": getattr(dataset, "SliceThickness", None),
    }

    return metadata