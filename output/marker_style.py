from __future__ import annotations

import math
from typing import Sequence, Tuple
from PIL import ImageDraw

ARROW_LENGTH_PX = 34.0
ARROW_HEAD_LENGTH_PX = 7.0
ARROW_HEAD_HALF_WIDTH_PX = 3.0
ARROW_LINE_WIDTH_PX = 2


def scaled_point(point: Sequence[float], original_shape, rendered_size: Tuple[int, int]):
    h, w = original_shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("invalid original image shape")
    return float(point[0]) * float(rendered_size[0]) / float(w), float(point[1]) * float(rendered_size[1]) / float(h)


def _choose_direction(x, y, width, height):
    dx = -1.0 if x >= ARROW_LENGTH_PX + 4 else 1.0
    dy = -1.0 if y >= ARROW_LENGTH_PX + 4 else 1.0
    if x > width - ARROW_LENGTH_PX - 4: dx = -1.0
    if y > height - ARROW_LENGTH_PX - 4: dy = -1.0
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def arrow_geometry(tip, canvas_size):
    x, y = float(tip[0]), float(tip[1])
    ux, uy = _choose_direction(x, y, canvas_size[0], canvas_size[1])
    start = (x + ux * ARROW_LENGTH_PX, y + uy * ARROW_LENGTH_PX)
    base_x, base_y = x + ux * ARROW_HEAD_LENGTH_PX, y + uy * ARROW_HEAD_LENGTH_PX
    px, py = -uy, ux
    head = ((x, y), (base_x + px * ARROW_HEAD_HALF_WIDTH_PX, base_y + py * ARROW_HEAD_HALF_WIDTH_PX), (base_x - px * ARROW_HEAD_HALF_WIDTH_PX, base_y - py * ARROW_HEAD_HALF_WIDTH_PX))
    return start, (x, y), head


def draw_precision_arrow(image, point, original_shape, fill="red"):
    x, y = scaled_point(point, original_shape, (image.width, image.height))
    start, tip, head = arrow_geometry((x, y), (image.width, image.height))
    draw = ImageDraw.Draw(image)
    draw.line([start, tip], fill=fill, width=ARROW_LINE_WIDTH_PX)
    draw.polygon(list(head), fill=fill)
    return tip
