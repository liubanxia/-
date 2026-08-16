import argparse

from core.runtime import PhoenixRuntime
from output.result_dispatcher import ResultDispatcher


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_ref",
        help="DICOM目录或PACS病例ID",
    )

    parser.add_argument(
        "--source",
        choices=["folder", "orthanc"],
        default="folder",
    )

    parser.add_argument(
        "--mode",
        choices=["A", "B"],
        default="A",
    )

    parser.add_argument(
        "--orthanc-url",
        default="http://127.0.0.1:8042",
    )

    args = parser.parse_args()

    phoenix = PhoenixRuntime()

    try:
        print("加载模型...")
        phoenix.load_models()

        print("\n模型状态：")
        for name, status in phoenix.model_hub.summary().items():
            print(f"{name}: {status}")

        kwargs = {}

        if args.source == "orthanc":
            kwargs["url"] = args.orthanc_url

        print("\n读取病例...")

        case = phoenix.open_case(
            args.source,
            args.case_ref,
            **kwargs,
        )

        print(
            f"读取到 {len(case.series)} 个序列"
        )

        print("开始AI分析...")

        result = phoenix.analyze()

        print(
            "调用模型：",
            result["selected_models"],
        )

        print(
            "病灶数量：",
            len(result["analysis"].lesions),
        )

        dispatcher = ResultDispatcher(
            args.mode
        )

        output = dispatcher.show(
            case,
            result,
            phoenix.memory,
        )

        if args.mode == "B":
            print(
                "PACS写回状态：",
                output,
            )

    finally:
        phoenix.shutdown()
        print("病例临时数据已清理")


if __name__ == "__main__":
    main()
