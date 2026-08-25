from __future__ import annotations

"""Run Phoenix unittest files in isolated interpreters.

Phoenix production runtime intentionally uses process-global monkey-patches.
Production/regression tests therefore run in their own processes with explicit
bootstrap. Low-level component tests run RAW so they validate the underlying
class contract rather than an intentionally different production routing layer.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_RAW_INFRASTRUCTURE_TESTS = {
    "test_blank_translation_student.py",
    "test_runtime_bootstrap_isolation.py",
    "test_sqlite_lifecycle_contract.py",
    "test_translation_api_value_ledger.py",
    "test_translation_learning_maturity_gate.py",
    "test_translation_learning_maturity_integration.py",
    "test_translation_survival_memory.py",
    # Component-level v2/base contracts. Production v3 routing is covered by
    # contextual/cascade/release tests and must not rewrite these unit semantics.
    "test_office_translation.py",
    "test_translation_backend_priority.py",
    "test_translation_product_v2.py",
}


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

    launcher = Path(__file__).with_name("run_one_unittest.py")
    for index, path in enumerate(files, start=1):
        production = path.name not in _RAW_INFRASTRUCTURE_TESTS
        mode = "PRODUCTION" if production else "RAW"
        print(
            f"\n===== [{index}/{len(files)}] {path.name} [{mode}] =====",
            flush=True,
        )
        command = [
            sys.executable,
            str(launcher),
            path.name,
            "--tests",
            str(root),
        ]
        if production:
            command.append("--bootstrap")
        completed = subprocess.run(command, env=env)
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
