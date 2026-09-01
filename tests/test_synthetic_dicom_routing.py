from types import SimpleNamespace

from core.case_router import ct_route_decision, get_xray_region, select_initial_models, select_models
from examples.synthetic_case.generate import create_synthetic_ct_series, create_synthetic_dr_image


def _case(modality, files, description=""):
    series = SimpleNamespace(
        modality=modality,
        files=[str(path) for path in files],
        description=description,
        series_description=description,
        study_description="SYNTHETIC STUDY",
        protocol_name=description,
        body_part="",
    )
    return SimpleNamespace(series=[series])


def test_synthetic_head_ct_routes_to_head_specialists(tmp_path):
    files = create_synthetic_ct_series(tmp_path / "head", body_part="HEAD", count=3)
    case = _case("CT", files)

    decision = ct_route_decision(case)
    models = select_models(case)

    assert decision["head"] is True
    assert decision["chest"] is False
    assert models[0] == "body_part_regression"
    assert "ich_2p5d_student" in models
    assert "brain_infarct_2p5d_student" in models
    assert "renal_stone_student" not in models


def test_router_result_adds_abdomen_specialists_without_private_config(tmp_path):
    files = create_synthetic_ct_series(tmp_path / "abdomen", body_part="ABDOMEN", count=2)
    case = _case("CT", files)

    models = select_models(case, {"active_body_regions": ["abdomen"]})

    assert models[0] == "body_part_regression"
    assert "renal_stone_student" in models
    assert "sbo_2p5d_student" in models
    assert "appendicitis_2p5d_student" in models


def test_router_text_fallback_recognizes_pelvis_without_active_regions(tmp_path):
    files = create_synthetic_ct_series(tmp_path / "pelvis", body_part="OTHER", count=2)
    case = _case("CT", files)

    router_result = {
        "active_body_regions": [],
        "body_part_display": "Pelvis",
    }
    decision = ct_route_decision(case, router_result)
    models = select_models(case, router_result)

    assert decision["pelvis"] is True
    assert decision["abdomen"] is False
    assert decision["head"] is False
    assert decision["chest"] is False
    assert "pelvis" in decision["router_regions"]
    assert models[0] == "body_part_regression"
    assert "renal_stone_student" in models
    assert "sbo_2p5d_student" in models
    assert "appendicitis_2p5d_student" in models


def test_synthetic_dr_chest_and_bone_routing(tmp_path):
    chest_path = create_synthetic_dr_image(tmp_path / "chest.dcm", body_part="CHEST")
    chest_case = _case("DR", [chest_path])
    assert get_xray_region(chest_case) == "chest"
    assert select_initial_models(chest_case) == ["chest_dr_nano_detector"]

    wrist_path = create_synthetic_dr_image(tmp_path / "wrist.dcm", body_part="WRIST")
    wrist_case = _case("DR", [wrist_path])
    assert get_xray_region(wrist_case) == "bone"
    assert select_initial_models(wrist_case) == [
        "fracture_rescbam",
        "fractureatlas_localization",
        "fractureatlas_segmentation",
    ]
