from core.runtime import PhoenixRuntime


def analyze_folder(case_folder, model_root=None):
    runtime = PhoenixRuntime(model_root=model_root)
    try:
        runtime.open_case("folder", case_folder)
        return runtime.analyze()
    finally:
        runtime.shutdown()
