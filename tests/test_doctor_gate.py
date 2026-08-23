import pytest

from ai.dual_vision_controller import DualVisionController


def test_inference_is_off_by_default():
    controller = DualVisionController()
    with pytest.raises(RuntimeError):
        controller.assert_inference_allowed()


def test_doctor_can_enable_then_context_change_resets():
    controller = DualVisionController()
    assert controller.start_by_doctor() is True
    assert controller.assert_inference_allowed() is True
    assert controller.reset_for_context_change() is True
    with pytest.raises(RuntimeError):
        controller.assert_inference_allowed()
