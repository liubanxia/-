from __future__ import annotations

"""Run Phoenix unittest files in isolated interpreters.

Phoenix production runtime intentionally uses process-global monkey-patches.
Production/regression tests therefore run in their own processes with explicit
bootstrap. Low-level component tests run RAW so they validate the underlying
class contract rather than an intentionally different production routing layer.
A small number of historical mixed-level files are split by test method: their
public product contracts run with production bootstrap while the explicit base
engine contract runs RAW.
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
    # Explicit component-level v2/base contracts. Production v3 routing is
    # covered by contextual/cascade/release tests.
    "test_translation_backend_priority.py",
    "test_translation_product_v2.py",
}

_OFFICE_BASE_METHOD = (
    "OfficeTranslationTests.test_batch_quality_model_retries_only_failed_segment_without_reasoning"
)
_STALE_V2_RELEASE_METHOD = (
    "ReleaseCandidateHardeningTests.test_failed_smart2_translation_never_falls_back_to_preview_model"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="tests", help="test directory")
    parser.add_argument("--pattern", default="test_*.py", help="glob pattern")
    return parser


def _command(launcher: Path, test_file: str, root: Path, *, bootstrap: bool, include="", exclude=""):
    command = [
        sys.executable,
        str(launcher),
        test_file,
        "--tests",
        str(root),
    ]
    if bootstrap:
        command.append("--bootstrap")
    if include:
        command.extend(("--include", include))
    if exclude:
        command.extend(("--exclude", exclude))
    return command


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
        print(f"\n===== [{index}/{len(files)}] {path.name} =====", flush=True)

        commands: list[tuple[str, list[str]]] = []
        if path.name == "test_office_translation.py":
            # Public Office publication/Workbench contracts are production v3.
            # One explicit MultiModelTranslationEngine batch unit remains RAW.
            commands.append((
                "PRODUCTION",
                _command(
                    launcher,
                    path.name,
                    root,
                    bootstrap=True,
                    exclude=_OFFICE_BASE_METHOD,
                ),
            ))
            commands.append((
                "RAW-BASE-ENGINE",
                _command(
                    launcher,
                    path.name,
                    root,
                    bootstrap=False,
                    include=_OFFICE_BASE_METHOD,
                ),
            ))
        elif path.name == "test_release_candidate_hardening.py":
            # This file contains one historical v2 expectation that an invalid
            # Smart2 result keeps the qwen backend label. v3 deliberately marks
            # that result as blocked_local_candidate instead. The replacement
            # safety contract lives in test_v3_candidate_blocking.py.
            commands.append((
                "PRODUCTION",
                _command(
                    launcher,
                    path.name,
                    root,
                    bootstrap=True,
                    exclude=_STALE_V2_RELEASE_METHOD,
                ),
            ))
        else:
            production = path.name not in _RAW_INFRASTRUCTURE_TESTS
            mode = "PRODUCTION" if production else "RAW"
            commands.append((
                mode,
                _command(
                    launcher,
                    path.name,
                    root,
                    bootstrap=production,
                ),
            ))

        failed = False
        for mode, command in commands:
            print(f"--- {path.name} [{mode}] ---", flush=True)
            completed = subprocess.run(command, env=env)
            if completed.returncode != 0:
                failed = True
        if failed:
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
