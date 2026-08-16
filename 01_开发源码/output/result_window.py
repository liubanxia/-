import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from output.lesion_viewer import LesionViewer


class ResultWindow:

    def show(self, result, memory=None):
        root = tk.Tk()
        root.title("Phoenix AI 辅助结果")
        root.geometry("650x560")

        analysis = result["analysis"]

        tk.Label(
            root,
            text=f"病例：{result['case_id']}",
            font=("Microsoft YaHei", 12, "bold"),
        ).pack(
            anchor="w",
            padx=12,
            pady=8,
        )

        tk.Label(
            root,
            text=f"病灶：{len(analysis.lesions)} 个",
        ).pack(
            anchor="w",
            padx=12,
        )

        if memory and memory.images:
            tk.Button(
                root,
                text="查看病灶",
                command=lambda:
                    LesionViewer().show(memory),
            ).pack(
                anchor="w",
                padx=12,
                pady=6,
            )

        text = ScrolledText(
            root,
            font=("Microsoft YaHei", 11),
        )
        text.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=8,
        )

        text.insert(
            "end",
            "【诊断结果】\n",
        )

        for item in analysis.diagnosis:
            text.insert(
                "end",
                f"- {item}\n",
            )

        text.insert(
            "end",
            "\n【报告草稿】\n",
        )

        text.insert(
            "end",
            analysis.report_draft,
        )

        text.configure(
            state="disabled"
        )

        root.mainloop()
