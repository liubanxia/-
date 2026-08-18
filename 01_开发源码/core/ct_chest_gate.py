import pydicom


CHEST_WORDS = (
    "CHEST",
    "THORAX",
    "LUNG",
    "CHEST CT",
    "胸",
    "肺",
)


def is_chest_ct(case):
    for series in getattr(
        case,
        "series",
        [],
    ):
        if str(
            getattr(series, "modality", "")
        ).upper() != "CT":
            continue

        values = []

        for attr in (
            "description",
            "series_description",
            "study_description",
            "body_part",
        ):
            value = getattr(
                series,
                attr,
                None,
            )

            if value:
                values.append(str(value))

        files = getattr(
            series,
            "files",
            [],
        ) or []

        if files:
            try:
                ds = pydicom.dcmread(
                    str(files[0]),
                    stop_before_pixels=True,
                    force=True,
                )

                for attr in (
                    "BodyPartExamined",
                    "StudyDescription",
                    "SeriesDescription",
                ):
                    value = getattr(
                        ds,
                        attr,
                        None,
                    )

                    if value:
                        values.append(str(value))

            except Exception:
                pass

        text = " ".join(values).upper()

        if any(
            word.upper() in text
            for word in CHEST_WORDS
        ):
            return True

    return False
