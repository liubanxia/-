from output.result_window import ResultWindow
from output.pacs_report_writer import VendorPacsWriter
from output.lesion_button import LesionButton


class ResultDispatcher:

    def __init__(self, mode="A", pacs_writer=None):
        self.mode = mode.upper()
        self.pacs_writer = pacs_writer
        self.lesion_button = LesionButton()

    def show(self, case, result, memory):
        if self.mode == "A":
            ResultWindow().show(
                result,
                memory,
            )
            return {"status": "shown"}

        if self.mode == "B":
            writer = self.pacs_writer or VendorPacsWriter()

            write_result = writer.write_report(
                case,
                result["analysis"].report_draft,
            )

            if memory and memory.images:
                self.lesion_button.start(
                    memory
                )

            return write_result

        raise ValueError(
            f"未知输出模式: {self.mode}"
        )

    def close(self):
        self.lesion_button.close()
