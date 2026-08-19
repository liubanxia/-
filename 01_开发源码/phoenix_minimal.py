import argparse

from core.environment_paths import get_environment_paths
from core.hardware_profile import detect_hardware_profile
from core.runtime import PhoenixRuntime
from output.result_dispatcher import ResultDispatcher


def _default_yunpacs_root():
    paths = get_environment_paths()
    if paths.image_root is not None:
        return str(paths.image_root)
    return "D:/YUNPACS/放射诊断/ImageDir_r"


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
        default=_default_yunpacs_root(),
        help="YUNPACS本地DICOM缓存目录",
    )

    args = parser.parse_args()
    env_paths = get_environment_paths()
    hardware = detect_hardware_profile()

    print(
        f"PHOENIX_ENV={env_paths.environment_name} "
        f"PROJECT_ROOT={env_paths.project_root}"
    )
    print(
        "PHOENIX_HARDWARE "
        f"mode={hardware.mode} "
        f"ram_gb={hardware.ram_gb} "
        f"gpu={hardware.gpu_name or '-'} "
        f"cuda={hardware.cuda_available} "
        f"device={hardware.inference_device} "
        f"heavy_3d_allowed={hardware.heavy_3d_allowed}"
    )
    print(f"PHOENIX_HARDWARE_REASON={hardware.reason}")

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

        print(f"CASE_ID={case.case_id}")
        print(f"STUDY_UID={getattr(case, 'study_uid', '')}")
        print(f"SOURCE_PATH={getattr(case, 'source_path', '')}")
        print(f"SERIES_COUNT={len(case.series)}")

        for warning in getattr(case, "warnings", []) or []:
            print(f"CASE_WARNING={warning}")

        print("开始AI分析...")

        result = phoenix.analyze()
        analysis = result["analysis"]

        print("调用模型：", result["selected_models"])
        print("病灶数量：", len(analysis.lesions))
        print(
            f"DIAGNOSTIC_EXECUTED={result.get('diagnostic_executed', False)}"
        )
        print(
            f"DIAGNOSTIC_VALID={result.get('diagnostic_valid', False)}"
        )
        print(
            "RESOLVED_LESION_GEOMETRY="
            f"{result.get('execution_summary', {}).get('resolved_lesion_geometry', 0)}"
        )

        print("\n模型真实执行结果：")

        executions = result.get(
            "execution_summary",
            {},
        ).get("models", [])

        execution_map = {
            item.get("model_name"): item
            for item in executions
            if isinstance(item, dict)
        }

        raw = analysis.raw_model_results

        for name in result["selected_models"]:
            item = execution_map.get(name, {})
            data = raw.get(name)

            if item:
                print(
                    f"{name}: "
                    f"status={item.get('status')} "
                    f"executed={item.get('executed')} "
                    f"processed={item.get('processed_images')} "
                    f"lesions={item.get('lesion_count')} "
                    f"device={item.get('device', '')} "
                    f"backend={item.get('backend', '')}"
                )
                if item.get("error"):
                    print(f"{name}: ERROR - {item.get('error')}")
                continue

            if data is None:
                print(f"{name}: 未执行")
            elif isinstance(data, dict) and "error" in data:
                print(f"{name}: ERROR - {data['error']}")
            else:
                print(f"{name}: 状态未知")

        dispatcher = ResultDispatcher(args.mode)

        output = dispatcher.show(
            case,
            result,
            phoenix.memory,
        )

        if args.mode == "B":
            print("PACS写回状态：", output)

    finally:
        phoenix.shutdown()
        print("病例临时数据已清理")


if __name__ == "__main__":
    main()
