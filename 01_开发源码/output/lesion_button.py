import tkinter as tk


class LesionButton:

    def show(self, overlays, callback=None):
        if not overlays:
            return

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        button = tk.Button(
            root,
            text="病灶",
            width=6,
            command=lambda: callback(overlays)
            if callback else None,
        )

        button.pack()

        root.geometry("+20+200")
        root.mainloop()
