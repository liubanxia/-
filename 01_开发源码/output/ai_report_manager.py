from output.ai_report_window import AIReportWindow


class AIReportManager:

    def __init__(self):
        self.window = None

    def show(self, controller, parent=None):
        if self.window is None:
            self.window = AIReportWindow(
                controller=controller,
                parent=parent,
            )
        else:
            self.window.set_controller(
                controller
            )

        # 当前病例已经有结构化初稿时，打开窗口立即显示。
        try:
            current = getattr(
                controller,
                "report_draft",
                None,
            )

            if current:
                self.window.set_report(current)
        except Exception:
            pass

        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

        return self.window

    def show_report(
        self,
        controller,
        text,
        parent=None,
    ):
        window = self.show(
            controller,
            parent
        )

        window.set_report(text)

        return window

    def clear_case(self):
        if self.window is None:
            return

        self.window.editor.clear()
        self.window.close()


AI_REPORT_MANAGER = AIReportManager()
