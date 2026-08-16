import tkinter as tk

from PIL import Image, ImageDraw, ImageTk


class LesionViewer:

    def show(self, memory):
        items = sorted(
            memory.images.values(),
            key=lambda x: x.get(
                "voxel_count",
                0,
            ),
            reverse=True,
        )

        if not items:
            return

        root = tk.Toplevel()
        root.title("病灶")
        root.attributes(
            "-topmost",
            True,
        )

        root.geometry("760x820")

        state = {
            "index": 0,
        }

        image_label = tk.Label(root)
        image_label.pack(
            padx=10,
            pady=10,
        )

        info = tk.Label(
            root,
            font=(
                "Microsoft YaHei",
                11,
            ),
        )
        info.pack(pady=6)

        def render():
            i = state["index"]
            item = items[i]

            image = Image.fromarray(
                item["image"]
            ).convert("RGB")

            original_shape = (
                item["image"].shape
            )

            image.thumbnail(
                (720, 680)
            )

            point = item.get(
                "point"
            )

            if point:
                self._draw_arrow(
                    image,
                    point,
                    original_shape,
                )

            photo = ImageTk.PhotoImage(
                image
            )

            image_label.configure(
                image=photo
            )

            image_label.image = photo

            info.configure(
                text=(
                    f"病灶 "
                    f"{i + 1}/{len(items)}"
                )
            )

        def previous():
            state["index"] = (
                state["index"] - 1
            ) % len(items)

            render()

        def next_item():
            state["index"] = (
                state["index"] + 1
            ) % len(items)

            render()

        buttons = tk.Frame(root)
        buttons.pack(pady=8)

        tk.Button(
            buttons,
            text="上一处",
            width=10,
            command=previous,
        ).pack(
            side="left",
            padx=8,
        )

        tk.Button(
            buttons,
            text="下一处",
            width=10,
            command=next_item,
        ).pack(
            side="left",
            padx=8,
        )

        render()

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

        x = int(
            point[0] * sx
        )

        y = int(
            point[1] * sy
        )

        start_x = max(
            0,
            x - 42,
        )

        start_y = max(
            0,
            y - 42,
        )

        draw.line(
            [
                (start_x, start_y),
                (x, y),
            ],
            fill="red",
            width=2,
        )

        # 很小的尖端，不画框、不覆盖病灶。
        draw.polygon(
            [
                (x, y),
                (x - 7, y - 2),
                (x - 2, y - 7),
            ],
            fill="red",
        )
