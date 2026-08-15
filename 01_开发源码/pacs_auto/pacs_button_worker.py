import gc
import threading
from pathlib import Path


class PacsAIWorker:
    def __init__(self):
        self.running = False
        self.result = None
        self.error = None

    def start(self):
        if self.running:
            return

        self.running = True

        threading.Thread(
            target=self._run,
            daemon=True
        ).start()

    def _run(self):
        import pythoncom
        pythoncom.CoInitialize()

        try:
            from pacs_auto import PacsAIAgent

            root = (
                Path(__file__).parents[2]
                / "04_AI模型"
                / "00_批量部署暂存"
                / "原始权重"
            )

            agent = PacsAIAgent(root)
            result = agent.run()

            status = result.get("status")

            # 转成现有Phoenix UI认识的格式
            if status == "dr_complete":
                self.result = {
                    "status": "success",
                    "ai": result["result"],
                }

            elif status == "ct_complete":
                self.result = {
                    "status": "ct_collected",
                    "slice_count":
                        result["slice_count"],
                }

            else:
                self.result = result

        except Exception as exc:
            self.error = (
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            self.running = False
            gc.collect()
            pythoncom.CoUninitialize()
