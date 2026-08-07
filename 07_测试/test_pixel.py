import sys
import unittest
from pathlib import Path

import numpy as np
from pydicom.data import get_testdata_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "01_开发源码"

sys.path.insert(0, str(SOURCE_ROOT))

from dicom.reader import read_dicom
from dicom.pixel import (
    get_pixel_array,
    ct_to_hu,
    apply_window,
    normalize_dx,
)


class TestPixelProcessing(unittest.TestCase):
    """
    Project Phoenix 像素处理基础回归测试。
    """

    @classmethod
    def setUpClass(cls):
        cls.ct_file = Path(get_testdata_file("CT_small.dcm"))
        cls.dx_file = PROJECT_ROOT / "07_测试" / "synthetic_DX.dcm"

        cls.ct_ds = read_dicom(cls.ct_file)
        cls.dx_ds = read_dicom(cls.dx_file)

    def test_01_raw_pixel_array(self):
        """原始 PixelData 必须可以读取。"""

        pixel_array = get_pixel_array(self.ct_ds)

        self.assertEqual(pixel_array.shape, (128, 128))

    def test_02_ct_to_hu(self):
        """CT 必须可以转换为 HU 矩阵。"""

        hu = ct_to_hu(self.ct_ds)

        self.assertEqual(hu.shape, (128, 128))
        self.assertEqual(hu.dtype, np.float32)
        self.assertTrue(np.isfinite(hu).all())

    def test_03_ct_window(self):
        """窗宽窗位输出必须为 8-bit 灰度图。"""

        hu = ct_to_hu(self.ct_ds)
        image = apply_window(
            hu,
            window_center=40,
            window_width=400,
        )

        self.assertEqual(image.shape, (128, 128))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 255)

    def test_04_dx_normalization(self):
        """DX（DR）必须可以标准化为 8-bit 灰度图。"""

        image = normalize_dx(self.dx_ds)

        self.assertEqual(image.shape, (64, 64))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.min(), 0)
        self.assertEqual(image.max(), 255)

    def test_05_invalid_window_width(self):
        """WindowWidth <= 0 必须拒绝。"""

        hu = ct_to_hu(self.ct_ds)

        with self.assertRaises(ValueError):
            apply_window(
                hu,
                window_center=40,
                window_width=0,
            )

    def test_06_wrong_modality(self):
        """CT 与 DX 专用处理函数不能互相混用。"""

        with self.assertRaises(ValueError):
            ct_to_hu(self.dx_ds)

        with self.assertRaises(ValueError):
            normalize_dx(self.ct_ds)


if __name__ == "__main__":
    unittest.main(verbosity=2)