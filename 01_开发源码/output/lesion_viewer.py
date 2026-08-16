import tkinter as tk

from PIL import Image, ImageDraw, ImageTk


class LesionViewer:

    def show(self, memory):
        items = list(
            memory.images.values()
        )

        if not items:
            return

        root = tk.Toplevel()
        root.title("Phoenix 病灶")

        frame = tk.Frame(root)
        frame.pack(padx=10, pady=10)

        photos = []

        for item in items:
            image = Image.fromarray(
                item["image"]
            ).convert("RGB")

            image.thumbnail((700, 700))

            point = item.get("point")

            if point:
                self._draw_arrow(
                    image,
                    point,
                    item["image"].shape,
                )

            photo = ImageTk.PhotoImage(
                image
            )
            photos.append(photo)

            label = tk.Label(
                frame,
                image=photo,
                text=item.get(
                    "label",
                    "病灶",
                ),
                compound="top",
            )
            label.pack(pady=8)

        root._phoenix_photos = photos

    def _draw_arrow(
        self,
        image,
        point,
        original_shape,
    ):
        draw = ImageDraw.Draw(image)

        h, w = original_shape[:2]

        sx = image.width / w
        sy = image.height / h

        x = int(point[0] * sx)
        y = int(point[1] * sy)

        start = (
            max(0, x - 35),
            max(0, y - 35),
        )

        draw.line(
            [start, (x, y)],
            fill="red",
            width=2,
        )

        draw.ellipse(
            [
                x - 2,
                y - 2,
                x + 2,
                y + 2,
            ],
            fill="red",
        )
