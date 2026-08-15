class PacsRuntime:
    def __init__(self, agent, fracture_ai):
        self.agent = agent
        self.fracture_ai = fracture_ai

    def run_dr(self):
        if not self.agent.can_control():
            raise RuntimeError("PACS未确认，禁止AI取图")

        image = self.agent.capture_viewport()

        try:
            return self.fracture_ai.infer(image)
        finally:
            image = None
