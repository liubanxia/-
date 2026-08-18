from pathlib import Path
import shutil
import time
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pacs_io.yunpacs_cache_adapter import YUNPACSLocalCacheAdapter
from pacs_io.yunpacs_watcher import YUNPACSWatcher


ROOT = Path("D:/project_phoenix")
SIM_ROOT = ROOT / "07_测试/yunpacs_sim"

adapter = YUNPACSLocalCacheAdapter(
    root=SIM_ROOT,
)

watcher = YUNPACSWatcher(
    adapter=adapter,
    poll_seconds=0.5,
)

print("===== CASE 1 =====")

case1 = watcher.wait_next_case()

print("Detected:", case1.directory)
print("Modality:", case1.modality)
print("Files:", case1.file_count)


print()
print("===== CREATE CASE 2 =====")

dx = ROOT / "07_测试/synthetic_DX.dcm"
case2_dir = SIM_ROOT / "2026-08-17/TEST_DX"
case2_dir.mkdir(parents=True, exist_ok=True)

if dx.exists():
    dst = case2_dir / "DX_001.dcm"
    shutil.copy(dx, dst)
else:
    # Fallback if synthetic_DX.dcm is unavailable.
    src = case1.series[0].files[0]
    dst = case2_dir / "CT_001.dcm"
    shutil.copy(src, dst)

# Make this case newer than CASE 1.
now = time.time()
dst.touch()
time.sleep(1.2)

print("Created:", dst)

print()
print("===== WAIT CASE 2 =====")

case2 = watcher.wait_next_case()

print("Detected:", case2.directory)
print("Modality:", case2.modality)
print("Files:", case2.file_count)

if case1.fingerprint == case2.fingerprint:
    print("FAIL: watcher did not detect change")
    raise SystemExit(1)

print()
print("WATCHER SWITCH PASSED")
