from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file")
    parser.add_argument("--tests", default="tests")
    parser.add_argument("--bootstrap", action="store_true")
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
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
