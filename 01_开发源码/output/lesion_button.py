import threading
import tkinter as tk

from output.lesion_viewer import LesionViewer


class LesionButton:

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def start(self, memory):
        if not memory or not memory.images:
            return

        self.close()
        self._stop.clear()

        self._thread = threading.Thread(
            target=self._run,
            args=(memory,),
            daemon=True,
        )
        self._thread.start()

    def _run(self, memory):
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        width, height = 72, 36
        x = root.winfo_screenwidth() - width - 25
        y = root.winfo_screenheight() // 2
        root.geometry(f"{width}x{height}+{x}+{y}")

        tk.Button(
            root,
            text="病灶",
            font=("Microsoft YaHei", 10),
            command=lambda: LesionViewer().show(memory),
        ).pack(fill="both", expand=True)

        def watch_case():
            if self._stop.is_set() or not memory.images:
                root.destroy()
                return
            root.after(250, watch_case)

        watch_case()
        root.mainloop()

    def close(self):
        self._stop.set()
