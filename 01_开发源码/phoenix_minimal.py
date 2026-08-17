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
        choices=["folder", "orthanc", "yunpacs"],
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

    parser.add_argument(
        "--yunpacs-root",
        default="D:/YUNPACS/放射诊断/ImageDir_r",
        help="YUNPACS本地DICOM缓存目录",
    )

    args = parser.parse_args()

    phoenix = PhoenixRuntime()

    try:
        kwargs = {}

        if args.source == "orthanc":
            kwargs["url"] = args.orthanc_url

        if args.source == "yunpacs":
            kwargs["root"] = args.yunpacs_root

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

        print("\n模型真实执行结果：")

        raw = result["analysis"].raw_model_results

        for name in result["selected_models"]:
            data = raw.get(name)

            if data is None:
                print(f"{name}: 未执行")
                continue

            if isinstance(data, dict) and "error" in data:
                print(
                    f"{name}: ERROR - {data['error']}"
                )
                continue

            if isinstance(data, dict):
                print(
                    f"{name}: "
                    f"processed={data.get('processed_images', '?')} "
                    f"lesions={len(data.get('lesions', []))}"
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
