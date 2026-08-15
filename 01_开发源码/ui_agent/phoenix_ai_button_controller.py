
from pathlib import Path
from threading import Thread, Lock
import traceback
import sys


class PhoenixAIButtonController:
    """
    Project Phoenix 医生触发型AI控制器

    强制规则：
    1. 打开病例 ≠ 启动AI。
    2. 只有医生点击“小人/AI按钮”后才允许推理。
    3. 推理线程只能由按钮事件创建。
    4. 同一时间只允许一个AI任务。
    5. 推理结束后自动释放模型。
    6. 结果保留给UI显示，但模型不常驻。
    """

    STATE_IDLE = "IDLE"
    STATE_CASE_READY = "CASE_READY"
    STATE_RUNNING = "RUNNING"
    STATE_COMPLETE = "COMPLETE"
    STATE_ERROR = "ERROR"

    def __init__(
        self,
        ai_models_dir,
        on_state_change=None,
        on_result=None,
        on_error=None,
    ):
        self.ai_models_dir = Path(ai_models_dir)

        if str(self.ai_models_dir) not in sys.path:
            sys.path.insert(
                0,
                str(self.ai_models_dir)
            )

        from dicom_inference_service import (
            PhoenixDICOMInferenceService
        )

        self.service = PhoenixDICOMInferenceService(
            self.ai_models_dir
        )

        self.on_state_change = on_state_change
        self.on_result = on_result
        self.on_error = on_error

        self.state = self.STATE_IDLE

        self.current_dicom = None
        self.current_result = None
        self.last_error = None

        self._thread = None
        self._lock = Lock()

    # --------------------------------------------------------
    # UI状态通知
    # --------------------------------------------------------
    def _set_state(self, state, message=""):
        self.state = state

        if callable(self.on_state_change):
            try:
                self.on_state_change(
                    state,
                    message
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # 医生在PACS中切换到一个病例
    #
    # 注意：
    # 这里只登记当前病例路径。
    # 绝对不加载模型、不启动AI。
    # --------------------------------------------------------
    def set_current_case(self, dicom_path):
        with self._lock:

            if self.state == self.STATE_RUNNING:
                raise RuntimeError(
                    "AI正在运行，不能切换病例。"
                )

            self.current_dicom = Path(
                dicom_path
            )

            if not self.current_dicom.exists():
                raise FileNotFoundError(
                    self.current_dicom
                )

            self.current_result = None
            self.last_error = None

            self._set_state(
                self.STATE_CASE_READY,
                "病例已就绪，等待医生点击AI"
            )

        return {
            "state": self.state,
            "ai_started": False,
            "model_loaded": False,
        }

    # --------------------------------------------------------
    # 医生点击“小人/AI按钮”
    #
    # 这是唯一允许启动推理的入口。
    # --------------------------------------------------------
    def click_ai_button(self):

        with self._lock:

            if self.current_dicom is None:
                raise RuntimeError(
                    "当前没有可供AI处理的DICOM病例。"
                )

            if self.state == self.STATE_RUNNING:
                raise RuntimeError(
                    "AI已经在运行，请勿重复点击。"
                )

            self.current_result = None
            self.last_error = None

            self._set_state(
                self.STATE_RUNNING,
                "医生已启动AI，正在分析"
            )

            # 线程只在医生按钮点击以后创建
            self._thread = Thread(
                target=self._run_after_doctor_click,
                name="PhoenixDoctorTriggeredAI",
                daemon=True
            )

            self._thread.start()

        return {
            "accepted": True,
            "state": self.STATE_RUNNING,
            "message": "AI已由医生主动启动"
        }

    # --------------------------------------------------------
    # 真正推理
    # --------------------------------------------------------
    def _run_after_doctor_click(self):

        opened = False

        try:
            # doctor_confirmed=True只存在于
            # 明确的按钮点击路径中。
            self.service.open_case(
                self.current_dicom,
                doctor_confirmed=True
            )

            opened = True

            result = (
                self.service.run_current_case()
            )

            self.current_result = result

            self._set_state(
                self.STATE_COMPLETE,
                "AI分析完成"
            )

            if callable(self.on_result):
                try:
                    self.on_result(result)
                except Exception:
                    pass

        except Exception as e:

            self.last_error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

            self._set_state(
                self.STATE_ERROR,
                str(e)
            )

            if callable(self.on_error):
                try:
                    self.on_error(
                        self.last_error
                    )
                except Exception:
                    pass

        finally:

            # 不让模型在病例分析结束后长期占用资源
            if opened:
                try:
                    self.service.close_case()
                except Exception:
                    pass

    # --------------------------------------------------------
    # 当前运行状态
    # --------------------------------------------------------
    def is_running(self):
        return (
            self.state
            == self.STATE_RUNNING
        )

    # --------------------------------------------------------
    # 获取已完成的结果
    # --------------------------------------------------------
    def get_result(self):
        return self.current_result

    # --------------------------------------------------------
    # 获取错误
    # --------------------------------------------------------
    def get_error(self):
        return self.last_error

    # --------------------------------------------------------
    # UI关闭/切病例时清理
    # --------------------------------------------------------
    def reset(self):

        if self.state == self.STATE_RUNNING:
            raise RuntimeError(
                "AI仍在运行，当前不能重置。"
            )

        try:
            if self.service.active:
                self.service.close_case()
        except Exception:
            pass

        self.current_dicom = None
        self.current_result = None
        self.last_error = None

        self._set_state(
            self.STATE_IDLE,
            "AI待机"
        )

    # --------------------------------------------------------
    # 仅测试/关闭程序时可等待线程结束
    # 正式UI一般不需要调用
    # --------------------------------------------------------
    def wait(self, timeout=None):

        thread = self._thread

        if thread is not None:
            thread.join(
                timeout=timeout
            )

        return self.state
