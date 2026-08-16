import pydicom


HEAD_WORDS = {
    "HEAD",
    "BRAIN",
    "CRANIUM",
    "SKULL",
    "颅",
    "脑",
    "头",
}


def is_head_ct(case):
    for series in case.series:
        if str(series.modality).upper() != "CT":
            continue

        for path in series.files[:3]:
            try:
                ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    force=True,
                )
            except Exception:
                continue

            text = " ".join([
                str(getattr(ds, "BodyPartExamined", "")),
                str(getattr(ds, "StudyDescription", "")),
                str(getattr(ds, "SeriesDescription", "")),
                str(getattr(ds, "ProtocolName", "")),
            ]).upper()

            if any(word in text for word in HEAD_WORDS):
                return True

    return False
