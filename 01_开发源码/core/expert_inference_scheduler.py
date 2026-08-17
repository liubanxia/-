import os
import gc
import threading
import torch
import torch.nn.functional as F

from core.expert_input_adapter import EXPERT_INPUT_ADAPTER
from core.expert_runtime_bridge import EXPERT_RUNTIME_BRIDGE
from core.expert_feature_memory import EXPERT_FEATURE_MEMORY


class ExpertInferenceScheduler:

    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.running = False

    def start(self, case_info, execution_plan):
        if self.running:
            return False

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._worker,
            args=(dict(case_info), dict(execution_plan)),
            daemon=True,
            name="PhoenixExpertWorker",
        )

        self.running = True
        self.thread.start()

        return True

    def _run_encoder(self, name, adapter, ct, dr, device):
        adapter.load(device)

        try:
            with torch.inference_mode():

                if name == "M3D-CLIP::vision_encoder":
                    return adapter.model.encode_image(
                        ct.to(device)
                    )

                if name == "13_MedSigLIP_448_ModelScope::vision_encoder":
                    size = int(
                        adapter.model.config.vision_config.image_size
                    )

                    x = F.interpolate(
                        dr,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ).repeat(1, 3, 1, 1)

                    return adapter.model.vision_model(
                        pixel_values=x.to(device)
                    ).last_hidden_state

                if name == "RAD-DINO-MAIRA2::vision_encoder":
                    size = getattr(
                        adapter.model.config,
                        "image_size",
                        518,
                    )

                    if isinstance(size, (list, tuple)):
                        h, w = int(size[-2]), int(size[-1])
                    else:
                        h = w = int(size)

                    x = F.interpolate(
                        dr,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).repeat(1, 3, 1, 1)

                    return adapter.model(
                        pixel_values=x.to(device)
                    ).last_hidden_state

        finally:
            adapter.unload()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return None

    def _worker(self, case_info, plan):
        modality = str(
            case_info.get("modality", "")
        ).upper()

        path = case_info.get("source_path")

        device = os.environ.get(
            "PHOENIX_EXPERT_DEVICE",
            "cpu",
        )

        try:
            ct = None
            dr = None

            if modality == "CT":
                ct = EXPERT_INPUT_ADAPTER.prepare_ct(path)

            elif modality in {"DX", "DR", "CR", "XR", "MG"}:
                dr = EXPERT_INPUT_ADAPTER.prepare_dr(path)

            for name in plan.get("encoders", []):
                if self.stop_event.is_set():
                    break

                adapter = EXPERT_RUNTIME_BRIDGE.resolve(name)

                if adapter is None:
                    continue

                try:
                    feature = self._run_encoder(
                        name,
                        adapter,
                        ct,
                        dr,
                        device,
                    )

                    if feature is not None:
                        EXPERT_FEATURE_MEMORY.put(
                            name,
                            feature,
                            modality=modality,
                            status="READY",
                        )

                except Exception as e:
                    EXPERT_FEATURE_MEMORY.set_meta(
                        f"error::{name}",
                        f"{type(e).__name__}: {e}",
                    )

            # CT定位结果已经存在时，串行继续做prompt分割。
            if modality == "CT" and ct is not None:
                try:
                    from core.prompt_segmentation_runner import (
                        PROMPT_SEGMENTATION_RUNNER,
                    )
                    from core.segmentation_result_bridge import (
                        SEGMENTATION_RESULT_BRIDGE,
                    )
                    from core.clinical_case_controller import (
                        CLINICAL_CASE_CONTROLLER,
                    )

                    seg_results = []
                    seg_results += PROMPT_SEGMENTATION_RUNNER.run_segvol(ct)
                    seg_results += PROMPT_SEGMENTATION_RUNNER.run_sam_med3d(ct)

                    seg_findings = SEGMENTATION_RESULT_BRIDGE.convert(
                        seg_results
                    )

                    if seg_findings:
                        CLINICAL_CASE_CONTROLLER.accept_expert_results(
                            seg_findings
                        )

                except Exception as e:
                    EXPERT_FEATURE_MEMORY.set_meta(
                        "segmentation_error",
                        f"{type(e).__name__}: {e}",
                    )

            EXPERT_FEATURE_MEMORY.set_meta(
                "pending_segmentation",
                list(plan.get("segmentation", [])),
            )

        finally:
            self.running = False
            gc.collect()

    def stop(self):
        self.stop_event.set()
        self.running = False


EXPERT_INFERENCE_SCHEDULER = ExpertInferenceScheduler()
