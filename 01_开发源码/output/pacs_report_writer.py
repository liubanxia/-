from abc import ABC, abstractmethod


class PacsReportWriter(ABC):

    @abstractmethod
    def write_report(self, case, report_text):
        pass


class VendorPacsWriter(PacsReportWriter):

    def __init__(self, adapter=None):
        self.adapter = adapter

    def write_report(self, case, report_text):
        if self.adapter is None:
            return {
                "status": "adapter_required",
                "case_id": case.case_id,
                "report": report_text,
            }

        return self.adapter.write_report(
            case,
            report_text,
        )
