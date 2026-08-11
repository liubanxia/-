class DualVisionController:
    """
    Project Phoenix 双视觉 AI 控制器。

    当前 M9.0-A 阶段只管理：
    - 医生主动启动
    - 医生主动停止
    - 双视觉运行状态
    - 推理前安全门控

    当前不加载任何真实 AI 模型。
    """

    def __init__(self):
        # --------------------------------------------------
        # 安全原则：
        # Phoenix 启动后，双视觉 AI 必须默认关闭。
        # --------------------------------------------------
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
        """
        仅供医生主动操作触发。

        后续 GUI 的“启动双视觉AI”按钮
        将调用本方法。
        """

        if self._active:
            return False

        self._active = True
        self._vision_a_active = True
        self._vision_b_active = True

        return True

    def stop_by_doctor(self):
        """
        医生主动停止双视觉 AI。
        """

        if not self._active:
            return False

        self._active = False
        self._vision_a_active = False
        self._vision_b_active = False

        return True

    def assert_inference_allowed(self):
        """
        所有真实视觉模型推理前必须经过此门控。

        未经医生主动启动时，
        禁止任何视觉 AI 推理。
        """

        if not self._active:
            raise RuntimeError(
                "双视觉AI未由医生主动启动，禁止推理"
            )

        if not (
            self._vision_a_active
            and self._vision_b_active
        ):
            raise RuntimeError(
                "双视觉AI状态不完整，禁止推理"
            )

        return True

    def reset_for_context_change(self):
        """
        病例、Study 或 Series 发生变化时调用。

        安全原则：
        新影像上下文不得继承上一病例的 AI 激活状态。
        无论当前是否正在运行，都强制恢复为默认关闭。
        """

        was_active = (
            self._active
            or self._vision_a_active
            or self._vision_b_active
        )

        self._active = False
        self._vision_a_active = False
        self._vision_b_active = False

        return was_active

    def get_status(self):
        """
        返回当前双视觉状态，
        后续可直接用于 UI 状态显示。
        """

        return {
            "dual_vision_active": self._active,
            "vision_a_active": self._vision_a_active,
            "vision_b_active": self._vision_b_active,
        }
