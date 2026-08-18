from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pacs_io.yunpacs_cache_adapter import (
    YUNPACSLocalCacheAdapter,
)

PROJECT = Path("D:/project_phoenix")

SOURCE = PROJECT / "07_测试/ct_series_test"

SIM_ROOT = PROJECT / "07_测试/yunpacs_sim"
SIM_CASE = SIM_ROOT / "2026-08-17/TEST_CT"

print("===== YUNPACS SIM TEST =====")
print("Source:", SOURCE)
print("Sim:", SIM_CASE)

if not SOURCE.exists():
    print("FAIL: CT test directory not found")
    raise SystemExit(1)

SIM_CASE.mkdir(
    parents=True,
    exist_ok=True,
)

files = sorted(
    p for p in SOURCE.iterdir()
    if p.is_file()
)

if not files:
    print("FAIL: no test files")
    raise SystemExit(2)

copied = 0

for src in files:
    dst = SIM_CASE / (
        src.name
        if src.suffix.lower() == ".dcm"
        else src.name + ".dcm"
    )

    shutil.copy2(
        src,
        dst,
    )
    copied += 1

print("Copied:", copied)

adapter = YUNPACSLocalCacheAdapter(
    root=SIM_ROOT,
)

case = adapter.latest_case(
    wait=True,
)

if case is None:
    print("FAIL: adapter did not find case")
    raise SystemExit(3)

print()
print("===== RESULT =====")
print("PASS")
print("Directory:", case.directory)
print("Modality:", case.modality)
print("Files:", case.file_count)
print("Series:", len(case.series))
print("StudyUID present:", bool(case.study_uid))

for i, series in enumerate(case.series, 1):
    print(
        f"Series {i}: "
        f"{series.modality} "
        f"images={len(series.files)} "
        f"description={series.description!r}"
    )

print()
print("YUNPACS SIMULATION PASSED")
