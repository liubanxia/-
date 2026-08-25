from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def _filtered_suite(suite, *, include: str = "", exclude: str = ""):
    selected = []
    for test in _iter_tests(suite):
        test_id = str(test.id())
        if include and include not in test_id:
            continue
        if exclude and exclude in test_id:
            continue
        selected.append(test)
    return unittest.TestSuite(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file")
    parser.add_argument("--tests", default="tests")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--include", default="")
    parser.add_argument("--exclude", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

    if args.bootstrap:
        import phoenix_knowledge

        phoenix_knowledge.bootstrap_runtime()

    tests_root = Path(args.tests).resolve()
    suite = unittest.defaultTestLoader.discover(
        str(tests_root),
        pattern=Path(args.test_file).name,
    )
    suite = _filtered_suite(
        suite,
        include=str(args.include or ""),
        exclude=str(args.exclude or ""),
    )
    count = suite.countTestCases()
    if count <= 0:
        print(
            f"ZERO_TESTS={Path(args.test_file).name} "
            f"include={args.include!r} exclude={args.exclude!r}",
            flush=True,
        )
        return 3

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
