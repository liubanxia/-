try:
    from pywinauto.uia_defines import IUIA
    from pywinauto.uia_element_info import UIAElementInfo
except ImportError:
    from pywinauto.windows.uia_defines import IUIA
    from pywinauto.windows.uia_element_info import UIAElementInfo

from pywinauto.controls.uiawrapper import UIAWrapper


class FocusedUIATextReader:
    """
    读取当前被医生选中的报告编辑控件。

    不监听键盘。
    不读取PASS数据库。
    只读取当前控件的完整文本。
    """

    def read_text(self):
        if self._bound_wrapper is not None:
            wrapper = self._bound_wrapper
        else:
            elem = IUIA().iuia.GetFocusedElement()
            info = UIAElementInfo(elem)
            wrapper = UIAWrapper(info)

        try:
            value = wrapper.iface_value.CurrentValue

            if value is not None:
                return str(value)

        except Exception:
            pass

        try:
            value = (
                wrapper.iface_text
                .DocumentRange
                .GetText(-1)
            )

            return str(value)

        except Exception:
            pass

        raise RuntimeError(
            "当前焦点控件不支持ValuePattern或TextPattern"
        )

    def __init__(self):
        self._bound_wrapper = None

    def bind_focused_control(self):
        """
        医生点击PASS报告编辑框后调用一次，
        Phoenix随后锁定这个控件。
        """
        elem = IUIA().iuia.GetFocusedElement()

        info = UIAElementInfo(elem)
        wrapper = UIAWrapper(info)

        self._bound_wrapper = wrapper

        return {
            "name": info.name,
            "control_type": info.control_type,
            "class_name": info.class_name,
            "automation_id": info.automation_id,
        }

    def clear_binding(self):
        self._bound_wrapper = None
