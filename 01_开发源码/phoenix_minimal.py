from core.runtime import PhoenixRuntime
from output.result_window import ResultWindow


def run_folder(case_path):
    phoenix = PhoenixRuntime()

    try:
        phoenix.load_models()

        phoenix.open_case(
            "folder",
            case_path,
        )

        result = phoenix.analyze()

        ResultWindow().show(
            result
        )

    finally:
        phoenix.shutdown()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "用法：python phoenix_minimal.py D:/CT_TEST"
        )
        raise SystemExit(1)

    run_folder(sys.argv[1])
