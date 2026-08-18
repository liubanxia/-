from pathlib import Path

CACHE = Path("D:/YUNPACS/放射诊断/ImageDir_r")

print("YUNPACS CACHE:", CACHE)

if not CACHE.exists():
    print("FAIL: cache not found")
    raise SystemExit(1)

cases = []

for p in CACHE.glob("*/*"):
    if p.is_dir():
        dcms = list(p.glob("*.dcm"))
        if dcms:
            newest = max(x.stat().st_mtime for x in dcms)
            cases.append((newest, p, len(dcms)))

if not cases:
    print("FAIL: no DICOM case found")
    raise SystemExit(2)

cases.sort(reverse=True)

_, path, count = cases[0]

print("PASS")
print("Latest case:", path)
print("DICOM count:", count)
