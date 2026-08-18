from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "01_开发源码"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PIL import Image

from output.marker_style import draw_precision_arrow, scaled_point


class MarkerPrecisionTest(unittest.TestCase):

    def test_arrow_tip_is_exact_scaled_target(self):
        image = Image.new("RGB", (512, 512), "black")
        original_shape = (1024, 1024)
        point = (256.0, 768.0)

        tip = draw_precision_arrow(
            image=image,
            point=point,
            original_shape=original_shape,
        )

        expected = scaled_point(
            point,
            original_shape,
            (image.width, image.height),
        )

        self.assertAlmostEqual(tip[0], expected[0], places=7)
        self.assertAlmostEqual(tip[1], expected[1], places=7)

    def test_edge_targets_are_not_shifted(self):
        image = Image.new("RGB", (512, 512), "black")
        original_shape = (512, 512)

        for point in (
            (2.0, 2.0),
            (509.0, 2.0),
            (2.0, 509.0),
            (509.0, 509.0),
        ):
            tip = draw_precision_arrow(
                image=image,
                point=point,
                original_shape=original_shape,
            )
            self.assertAlmostEqual(tip[0], point[0], places=7)
            self.assertAlmostEqual(tip[1], point[1], places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
