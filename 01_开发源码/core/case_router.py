XRAY_MODALITIES = {
    "DX", "DR", "CR", "XR",
}


def get_modalities(case):
    return {
        str(series.modality).upper()
        for series in case.series
    }


def select_models(case):
    modalities = get_modalities(case)

    if "CT" in modalities:
        return [
            "body_part_regression",
            "blast_ct_head",
        ]

    if modalities & XRAY_MODALITIES:
        return [
            "fracture_rescbam",
            "fractureatlas_localization",
            "fractureatlas_segmentation",
        ]

    return []
