from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pacs_io.yunpacs_cache_adapter import YUNPACSLocalCacheAdapter
from pacs_io.yunpacs_bridge import YUNPACSPhoenixBridge


ROOT = Path("D:/project_phoenix")
SIM_ROOT = ROOT / "07_测试/yunpacs_sim"

adapter = YUNPACSLocalCacheAdapter(root=SIM_ROOT)

ct_dir = SIM_ROOT / "2026-08-17/TEST_CT"
dx_dir = SIM_ROOT / "2026-08-17/TEST_DX"

ct = adapter.build_case(ct_dir, wait=False)
dx = adapter.build_case(dx_dir, wait=False)

events = []


def on_case(case):
    events.append(("OPEN", case.modality, case.directory.name))
    print(
        "OPEN:",
        case.modality,
        case.directory.name,
        case.file_count,
    )


def on_close(case):
    events.append(("CLOSE", case.modality, case.directory.name))
    print(
        "CLOSE:",
        case.modality,
        case.directory.name,
    )


bridge = YUNPACSPhoenixBridge(
    on_case=on_case,
    on_case_close=on_close,
)

print("===== OPEN CT =====")
bridge.handle_case(ct)

print()
print("===== SWITCH TO DX =====")
bridge.handle_case(dx)

print()
print("===== VERIFY =====")

if len(events) != 3:
    print("FAIL: unexpected event count:", events)
    raise SystemExit(1)

if events[0][0] != "OPEN":
    raise SystemExit("FAIL: CT did not open")

if events[1][0] != "CLOSE":
    raise SystemExit("FAIL: previous case did not close")

if events[2][0] != "OPEN":
    raise SystemExit("FAIL: DX did not open")

if bridge.current_case.study_uid != dx.study_uid:
    raise SystemExit("FAIL: current case is not DX")

print("BRIDGE LIFECYCLE PASSED")
