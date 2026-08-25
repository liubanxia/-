from __future__ import annotations

"""Run Phoenix unittest files in isolated interpreters.

Phoenix production runtime intentionally uses process-global monkey-patches.
Running every test module inside one unittest-discovery interpreter allows a GUI
or production-bootstrap test to mutate classes used by later low-level tests.
That creates order-dependent false failures and can hide real isolation bugs.

This runner preserves unittest discovery within each file while giving every
file a fresh Python process and environment.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="tests", help="test directory")
    parser.add_argument("--pattern", default="test_*.py", help="glob pattern")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(args.tests).resolve()
    files = sorted(path for path in root.glob(args.pattern) if path.is_file())
    if not files:
        print(f"NO_TEST_FILES={root / args.pattern}")
        return 2

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("PYTHONUTF8", "1")
    failures: list[str] = []

    for index, path in enumerate(files, start=1):
        print(f"\n===== [{index}/{len(files)}] {path.name} =====", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(root),
                "-p",
                path.name,
                "-v",
            ],
            env=env,
        )
        if completed.returncode != 0:
            failures.append(path.name)

    print("\n===== ISOLATED TEST SUMMARY =====")
    print(f"FILES={len(files)}")
    print(f"FAILED_FILES={len(failures)}")
    if failures:
        for name in failures:
            print(f"FAIL={name}")
        return 1
    print("ALL_ISOLATED_TEST_FILES=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
