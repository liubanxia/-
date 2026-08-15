class PacsToolController:
    def __init__(self, agent):
        self.agent = agent
        self.profile = {}

    def probe_tools(self):
        self.agent._guard()

        tests = {
            "右键水平拖动": ("right", 30, 0),
            "右键垂直拖动": ("right", 0, 30),
            "左键水平拖动": ("left", 30, 0),
            "左键垂直拖动": ("left", 0, 30),
            "中键水平拖动": ("middle", 30, 0),
        }

        results = {}

        for name, (button, dx, dy) in tests.items():
            try:
                result = self.agent.probe_drag(
                    button=button,
                    dx=dx,
                    dy=dy,
                )
                results[name] = result
            except Exception as e:
                results[name] = {
                    "error": str(e)
                }

        self.profile = results
        return results

    def scroll_slice(self, direction):
        self.agent._guard()
        return self.agent.safe_scroll(direction)

    def drag(self, button, dx, dy):
        self.agent._guard()
        return self.agent.safe_drag(
            button=button,
            dx=dx,
            dy=dy,
        )
