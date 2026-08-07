import sys
import unittest
from pathlib import Path

from pydicom.data import get_testdata_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "01_开发源码"

sys.path.insert(0, str(SOURCE_ROOT))

from dicom.reader import read_dicom
from dicom.metadata import extract_metadata


class TestDicomReader(unittest.TestCase):
    """
    Project Phoenix DICOM Reader 基础回归测试。

    当前验证范围：
    - CT 必须允许
    - DX（DR）必须允许
    - MR 必须拒绝
    - 不存在文件必须拒绝
    - 非 DICOM 文件必须拒绝
    - CT / DX PixelData 必须可以读取
    """

    @classmethod
    def setUpClass(cls):
        cls.ct_file = Path(get_testdata_file("CT_small.dcm"))
        cls.mr_file = Path(get_testdata_file("MR_small.dcm"))
        cls.dx_file = PROJECT_ROOT / "07_测试" / "synthetic_DX.dcm"

    def test_01_ct_should_be_accepted(self):
        """CT 必须允许读取。"""

        ds = read_dicom(self.ct_file)

        self.assertEqual(ds.Modality, "CT")
        self.assertEqual(ds.Rows, 128)
        self.assertEqual(ds.Columns, 128)

    def test_02_ct_pixel_data_should_be_readable(self):
        """CT PixelData 必须可以转换为像素矩阵。"""

        ds = read_dicom(self.ct_file)
        pixel_array = ds.pixel_array

        self.assertEqual(pixel_array.shape, (128, 128))

    def test_03_ct_metadata_should_be_available(self):
        """CT 基础元数据必须可以提取。"""

        ds = read_dicom(self.ct_file)
        metadata = extract_metadata(ds)

        self.assertEqual(metadata["Modality"], "CT")
        self.assertEqual(metadata["Rows"], 128)
        self.assertEqual(metadata["Columns"], 128)

    def test_04_dx_should_be_accepted(self):
        """DX（DR）必须允许读取。"""

        self.assertTrue(
            self.dx_file.exists(),
            "synthetic_DX.dcm 不存在，请先运行 create_dx_test.py",
        )

        ds = read_dicom(self.dx_file)

        self.assertEqual(ds.Modality, "DX")
        self.assertEqual(ds.Rows, 64)
        self.assertEqual(ds.Columns, 64)

    def test_05_dx_pixel_data_should_be_readable(self):
        """DX PixelData 必须可以转换为像素矩阵。"""

        ds = read_dicom(self.dx_file)
        pixel_array = ds.pixel_array

        self.assertEqual(pixel_array.shape, (64, 64))

    def test_06_mr_should_be_rejected(self):
        """V1 不支持 MR，必须明确拒绝。"""

        with self.assertRaises(ValueError):
            read_dicom(self.mr_file)

    def test_07_missing_file_should_be_rejected(self):
        """不存在的文件必须拒绝。"""

        missing_file = (
            PROJECT_ROOT
            / "07_测试"
            / "file_that_does_not_exist.dcm"
        )

        with self.assertRaises(FileNotFoundError):
            read_dicom(missing_file)

    def test_08_invalid_dicom_should_be_rejected(self):
        """普通文本伪装成 DICOM 时必须拒绝。"""

        invalid_file = PROJECT_ROOT / "07_测试" / "invalid_test.dcm"

        invalid_file.write_text(
            "THIS IS NOT A DICOM FILE",
            encoding="utf-8",
        )

        try:
            with self.assertRaises(ValueError):
                read_dicom(invalid_file)
        finally:
            if invalid_file.exists():
                invalid_file.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)