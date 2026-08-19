from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.environment_paths import resolve_project_root


M = "04_AI模型"
B = f"{M}/批量专家池"
CTSEG = f"{B}/CT_分割"
MONAI = f"{B}/MONAI_CT专科"
SUPREM = f"{B}/CT_预训练竞赛/SuPreM_family"
TEACHER = f"{M}/教师模型"
CANDIDATE = f"{M}/教师候选池"

WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".bin", ".safetensors"}

# (delete_path, reason, canonical_path_or_None)
SAFE = [
    (f"{CTSEG}/NV-Segment-CT/vista3d_pretrained_model", "duplicate VISTA3D", f"{CTSEG}/VISTA3D-HF/vista3d_pretrained_model"),
    (f"{CTSEG}/NV-Segment-CTMR/vista3d_pretrained_model", "duplicate VISTA3D", f"{CTSEG}/VISTA3D-HF/vista3d_pretrained_model"),
    (f"{CTSEG}/SegVol-ModelScope", "duplicate SegVol", f"{CTSEG}/SegVol"),
    (f"{TEACHER}/04_SegVol_ModelScope", "duplicate SegVol", f"{CTSEG}/SegVol"),
    (f"{SUPREM}/supervised_dodnet_unet_920.pth", "redundant SuPreM", None),
    (f"{SUPREM}/supervised_suprem_unet_2100.pth", "redundant SuPreM", None),
    (f"{SUPREM}/supervised_clip_driven_universal_unet_2100.pth", "redundant SuPreM", None),
    (f"{SUPREM}/supervised_suprem_swinunetr_2100.pth", "redundant large SuPreM", None),
    (f"{SUPREM}/supervised_clip_driven_universal_swin_unetr_2100.pth", "redundant large SuPreM", None),
    (f"{SUPREM}/supervised_med3D_residual_unet_1623.pth", "redundant SuPreM", None),
    (f"{SUPREM}/self_supervised_models_genesis_unet_620.pt", "redundant SuPreM", None),
    (f"{MONAI}/multi_organ_segmentation", "overlaps canonical whole-body teacher", None),
    (f"{MONAI}/renalStructures_UNEST_segmentation", "heavier renal duplicate", None),
    (f"{MONAI}/spleen_ct_segmentation", "covered by whole-body teacher", None),
    (f"{MONAI}/wholeBrainSeg_Large_UNEST_segmentation", "MRI-oriented model in CT pool", None),
    (f"{MONAI}/pediatric_abdominal_ct_segmentation", "outside current adult target", None),
    (f"{M}/待接入模型/SAM-Med3D", "generic segmentation overlap", None),
    (f"{M}/待接入模型/MedSAM2/checkpoints/MedSAM2_latest.pt", "keep CTLesion plus tiny base only", None),
    (f"{B}/CT_通用/Merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt", "redundant Merlin encoder", None),
    (f"{B}/CT_通用/Merlin/nnUNetTrainerMerlin__nnUNetPlans__3d_fullres", "redundant Merlin encoder tree", None),
    (f"{CANDIDATE}/通用/MedGemma-1.5-4B", "duplicate smaller reasoning teacher", f"{TEACHER}/11_MedGemma_1.5_4B_ModelScope"),
    (f"{TEACHER}/14_MAIRA_2_ModelScope", "duplicate full report stack", f"{TEACHER}/15_RAD_DINO_MAIRA2_ModelScope"),
    (f"{M}/待接入模型/TotalSegmentator-weights", "empty/incomplete weight cache", None),
]

AGGRESSIVE = [
    (f"{CANDIDATE}/重量级/Lingshu-32B", "oversized language-only package"),
    (f"{CANDIDATE}/重量级/MedGemma-27B", "oversized; smaller reasoning teacher retained"),
    (f"{TEACHER}/12_MedGemma_27B_ModelScope", "oversized duplicate teacher"),
    (f"{CANDIDATE}/通用/HealthGPT-Pro-8B", "redundant language teacher"),
    (f"{CANDIDATE}/通用/Lingshu-7B", "redundant language teacher"),
    (f"{CANDIDATE}/通用/Fleming-VL-8B", "unvalidated redundant VLM"),
    (f"{CANDIDATE}/通用/Lingshu-I-8B", "unvalidated redundant VLM"),
    (f"{CANDIDATE}/医学视觉/LLaVA-Med-7B", "unvalidated redundant medical VLM"),
    (f"{M}/待接入模型/Hulu-Med-4B", "unvalidated redundant medical VLM"),
    (f"{CANDIDATE}/通用/HealthGPT-Pro-4B", "redundant language teacher"),
    (f"{CANDIDATE}/通用/MedGemma-4B-old", "old duplicate MedGemma"),
]

PROTECTED = {
    f"{M}/路由模型/BodyPartRegression",
    f"{M}/视觉B_骨折防护/YOLOv8_ResCBAM.onnx",
    f"{M}/00_批量部署暂存/原始权重/yolov8_localization_fractureAtlas.pt",
    f"{M}/00_批量部署暂存/原始权重/yolov8_segmentation_fractureAtlas.pt",
    f"{CTSEG}/VISTA3D-HF",
    f"{CTSEG}/SegVol",
    f"{M}/待接入模型/MedSAM2/checkpoints/MedSAM2_CTLesion.pt",
    f"{M}/待接入模型/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt",
    f"{B}/CT_通用/M3D-CLIP",
    f"{MONAI}/wholeBody_ct_segmentation",
    f"{MONAI}/pancreas_ct_dints_segmentation",
    f"{MONAI}/renalStructures_CECT_segmentation",
    f"{MONAI}/lung_nodule_ct_detection",
    f"{TEACHER}/11_MedGemma_1.5_4B_ModelScope",
    f"{TEACHER}/13_MedSigLIP_448_ModelScope",
    f"{TEACHER}/15_RAD_DINO_MAIRA2_ModelScope",
}


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} TB"


def has_weight(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in WEIGHT_SUFFIXES and path.stat().st_size > 0
    if not path.exists():
        return False
    return any(
        x.is_file() and x.suffix.lower() in WEIGHT_SUFFIXES and x.stat().st_size > 0
        for x in path.rglob("*")
    )


def protected(rel: str) -> bool:
    return any(
        rel == item or rel.startswith(item + "/") or item.startswith(rel + "/")
        for item in PROTECTED
    )


def has_lfs_pointer(repo_root: Path) -> bool:
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        try:
            if path.stat().st_size <= 2048 and b"git-lfs.github.com/spec/v1" in path.read_bytes()[:256]:
                return True
        except OSError:
            pass
    return False


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--aggressive", action="store_true")
    ap.add_argument("--purge-model-git-cache", action="store_true")
    ap.add_argument("--keep", action="append", default=[])
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = resolve_project_root(args.root)
    candidates = list(SAFE)
    if args.aggressive:
        candidates += [(p, r, None) for p, r in AGGRESSIVE]

    plan, skipped = [], []
    for rel, reason, canonical in candidates:
        path = root / rel
        if not path.exists():
            continue
        if protected(rel) or any(k.lower() in rel.lower() for k in args.keep):
            skipped.append({"path": rel, "reason": "protected"})
            continue
        if canonical and not has_weight(root / canonical):
            skipped.append({"path": rel, "reason": f"canonical not verified: {canonical}"})
            continue
        plan.append({"path": rel, "bytes": size_bytes(path), "reason": reason})

    for path in root.rglob("*"):
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
            rel = path.relative_to(root).as_posix()
            plan.append({"path": rel, "bytes": size_bytes(path), "reason": "generated cache"})

    if args.purge_model_git_cache:
        model_root = root / M
        if model_root.exists():
            for gitdir in model_root.rglob(".git"):
                if not gitdir.is_dir():
                    continue
                rel = gitdir.relative_to(root).as_posix()
                if has_lfs_pointer(gitdir.parent):
                    skipped.append({"path": rel, "reason": "LFS pointers still present"})
                    continue
                plan.append({"path": rel, "bytes": size_bytes(gitdir), "reason": "nested Git/LFS cache"})

    plan.sort(key=lambda x: (x["path"].count("/"), x["path"]))
    final = []
    for item in plan:
        if any(item["path"] == x["path"] or item["path"].startswith(x["path"] + "/") for x in final):
            continue
        final.append(item)

    reclaim = sum(x["bytes"] for x in final)
    manifest = {
        "project_root": str(root),
        "mode": "apply" if args.apply else "dry_run",
        "aggressive": args.aggressive,
        "purge_model_git_cache": args.purge_model_git_cache,
        "estimated_reclaim_bytes": reclaim,
        "items": final,
        "skipped": skipped,
        "deleted": [],
        "errors": [],
    }
    log_dir = root / "06_日志"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"model_storage_cleanup_{datetime.now():%Y%m%d_%H%M%S}.json"
    log.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"PHOENIX_ROOT={root}")
    print(f"MODE={'APPLY' if args.apply else 'DRY_RUN'}")
    print(f"DELETE_ITEMS={len(final)}")
    print(f"ESTIMATED_RECLAIM={human(reclaim)}")
    print(f"MANIFEST={log}")
    for item in final:
        print(f"DELETE_CANDIDATE {human(item['bytes'])} {item['path']} :: {item['reason']}")
    for item in skipped:
        print(f"SKIP {item['path']} :: {item['reason']}")

    if not args.apply:
        return 0

    for item in final:
        path = root / item["path"]
        try:
            remove(path)
            manifest["deleted"].append(item["path"])
            print(f"DELETED={item['path']}")
        except Exception as exc:
            manifest["errors"].append({"path": item["path"], "error": repr(exc)})
            print(f"DELETE_FAILED={item['path']} :: {exc!r}")

    log.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DELETION_COMPLETE={len(manifest['deleted'])}/{len(final)}")
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
