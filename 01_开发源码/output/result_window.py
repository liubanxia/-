import tkinter as tk
from tkinter.scrolledtext import ScrolledText


class ResultWindow:

    def show(self, result):
        root = tk.Tk()
        root.title("Phoenix AI 辅助结果")
        root.geometry("620x520")

        analysis = result["analysis"]

        tk.Label(
            root,
            text=f"病例：{result['case_id']}",
            font=("Microsoft YaHei", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=8)

        text = ScrolledText(
            root,
            font=("Microsoft YaHei", 11),
        )
        text.pack(fill="both", expand=True, padx=12, pady=8)

        text.insert("end", "【诊断结果】\n")
        for item in analysis.diagnosis:
            text.insert("end", f"- {item}\n")

        text.insert("end", "\n【报告草稿】\n")
        text.insert("end", analysis.report_draft)

        text.configure(state="disabled")

        root.mainloop()
