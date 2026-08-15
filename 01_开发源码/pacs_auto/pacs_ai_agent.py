import re

from .pacs_auto_agent import PacsAutoAgent
from .pacs_ct_collector import PacsCTCollector
from .pacs_fracture_ai import PacsFractureAI
from .pacs_runtime import PacsRuntime
from .pacs_tool_controller import PacsToolController


class PacsAIAgent:

    def __init__(self, model_root):
        self.model_root = model_root
        self.state = "IDLE"

    def run(self):
        agent = PacsAutoAgent()

        self.state = "DISCOVER_PACS"
        pacs = agent.bind_best()

        if not agent.can_control():
            self.state = "WAIT_CONFIRM"
            return {
                "status": "needs_confirmation",
                "pacs": pacs,
            }

        # PACS工具自动识别/学习
        tools = PacsToolController(agent)

        named_tools = agent.find_tools()

        self.state = "LEARN_TOOLS"

        try:
            tool_profile = tools.probe_tools()
        except Exception:
            tool_profile = {}

        texts = []
        for c in agent.window.descendants():
            try:
                t = c.window_text().strip()
                if t:
                    texts.append(t)
            except Exception:
                pass

        text = " ".join(texts).upper()

        # CT
        if re.search(r"\bCT\b", text):
            self.state = "CT_READING"

            collector = PacsCTCollector(agent)
            count = collector.scan_full_ct()

            self.state = "DONE"

            return {
                "status": "ct_complete",
                "slice_count": count,
                "note": "CT自动翻层完成，尚未接入CT病灶诊断模型",
                "named_tools": named_tools,
                "tool_profile": tool_profile,
            }

        # DR / DX / CR
        if re.search(r"\b(DR|DX|CR|XR)\b|X[- ]?RAY", text):
            self.state = "DR_AI"

            ai = PacsFractureAI(
                self.model_root
            )

            runtime = PacsRuntime(
                agent,
                ai
            )

            result = runtime.run_dr()

            self.state = "DONE"

            return {
                "status": "dr_complete",
                "result": result,
                "named_tools": named_tools,
                "tool_profile": tool_profile,
            }

        self.state = "STOPPED"

        return {
            "status": "modality_unknown",
            "reason": "无法安全确认CT/DR，未执行AI",
        }
