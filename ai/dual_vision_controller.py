class DualVisionController:
    """Doctor-triggered safety gate for dual visual-AI inference."""

    def __init__(self):
        self._active = False
        self._vision_a_active = False
        self._vision_b_active = False

    @property
    def is_active(self):
        return self._active

    @property
    def vision_a_active(self):
        return self._vision_a_active

    @property
    def vision_b_active(self):
        return self._vision_b_active

    def start_by_doctor(self):
        if self._active:
            return False
        self._active = True
        self._vision_a_active = True
        self._vision_b_active = True
        return True

    def stop_by_doctor(self):
        if not self._active:
            return False
        self._active = False
        self._vision_a_active = False
        self._vision_b_active = False
        return True

    def assert_inference_allowed(self):
        if not self._active:
            raise RuntimeError("双视觉AI未由医生主动启动，禁止推理")
        if not (self._vision_a_active and self._vision_b_active):
            raise RuntimeError("双视觉AI状态不完整，禁止推理")
        return True

    def reset_for_context_change(self):
        was_active = self._active or self._vision_a_active or self._vision_b_active
        self._active = False
        self._vision_a_active = False
        self._vision_b_active = False
        return was_active

    def get_status(self):
        return {
            "dual_vision_active": self._active,
            "vision_a_active": self._vision_a_active,
            "vision_b_active": self._vision_b_active,
        }
