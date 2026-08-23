from output.result_window import ResultWindow


class ResultDispatcher:
    """Public result dispatcher. Vendor-specific PACS write-back is intentionally excluded."""

    def show(self, case, result, memory=None):
        ResultWindow().show(result, memory)
        return {"status": "shown"}
