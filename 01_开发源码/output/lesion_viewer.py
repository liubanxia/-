import tkinter as tk

from PIL import Image, ImageTk

from output.marker_style import draw_precision_arrow


class LesionViewer:

    def show(self, memory):
        items = sorted(
            memory.images.values(),
            key=lambda x: (
                float(x.get("confidence", 0) or 0),
                int(x.get("voxel_count", 0) or 0),
            ),
            reverse=True,
        )

        if not items:
            return

        root = tk.Toplevel()
        root.title("病灶定位")
        root.attributes(
            "-topmost",
            True,
        )
        root.geometry("780x850")

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
                10,
            ),
            justify="left",
            anchor="w",
        )
        info.pack(
            fill="x",
            padx=16,
            pady=6,
        )

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
                (740, 690)
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

            confidence = item.get(
                "confidence"
            )

            try:
                confidence_text = (
                    f"{float(confidence):.3f}"
                )
            except Exception:
                confidence_text = "N/A"

            info.configure(
                text=(
                    f"病灶 {i + 1}/{len(items)}\n"
                    f"候选：{item.get('label', '异常候选灶')}\n"
                    f"模型：{item.get('source_model', '')}\n"
                    f"SeriesUID：{item.get('series_uid', '')}\n"
                    f"层号：{item.get('image_index', '')}\n"
                    f"置信度：{confidence_text}"
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
        return draw_precision_arrow(
            image=image,
            point=point,
            original_shape=original_shape,
            fill="red",
        )
