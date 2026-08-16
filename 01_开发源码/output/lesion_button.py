import tkinter as tk

from output.lesion_viewer import LesionViewer


class LesionButton:

    def __init__(self):
        self.root = None

    def show(self, memory):
        if not memory or not memory.images:
            return

        root = tk.Tk()
        self.root = root

        root.title("Phoenix")
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        width = 72
        height = 36

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        x = screen_w - width - 25
        y = screen_h // 2

        root.geometry(
            f"{width}x{height}+{x}+{y}"
        )

        button = tk.Button(
            root,
            text="病灶",
            font=("Microsoft YaHei", 10),
            command=lambda: LesionViewer().show(memory),
        )

        button.pack(
            fill="both",
            expand=True,
        )

        def watch_case():
            if not memory.images:
                root.destroy()
                return

            root.after(
                500,
                watch_case,
            )

        watch_case()
        root.mainloop()
