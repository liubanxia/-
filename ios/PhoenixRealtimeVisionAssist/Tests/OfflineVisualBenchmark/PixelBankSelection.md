# Pixel capability bank selection

The old near-1GiB bank is not discarded. It is pruned by unique visual value rather than by size.

## Keep

- yolo11n — smallest localization probe.
- YOLOv3TinyInt8LUT — independent tiny localization fallback.
- YOLOv3Int8LUT — larger localization model for offline high-recall comparison.
- YOLOv3FP16 — higher-precision localization comparison.
- DeepLabV3Int8LUT — pixel-level foreground/region segmentation comparison.
- DeepLabV3FP16 — higher-precision pixel segmentation comparison.
- Resnet50Headless — generic feature embedding baseline.
- FastViTT8F16Headless — lightweight feature embedding.
- FastViTMA36F16Headless — higher-capacity feature embedding reserve.
- DepthAnythingV2SmallF16P6 — depth/geometry reserve.
- DepthAnythingV2SmallF16 — depth/geometry comparison.

## Remove from the rebuilt bank

- Full FP32 duplicates when an FP16 or Int8 variant of the same role is already retained.
- MobileNetV2 classification variants: ImageNet class prediction does not add useful localization pixels.
- ResNet50 classification variants: headless embedding is the useful part for visual comparison.
- FastViT classification variants: headless embeddings retain the representation capability without redundant classifier heads.
- Duplicate Tiny precision variants that do not provide a distinct role.

## Pixel rule

Never judge small-human recognition from a single whole-frame 384x384 resize. Offline evaluation must compare original-frame crops using `PixelScalePlanner`: full frame first, then overlapping tiles and high-pixel center regions. Results are fused only after mapping detections back to source-image coordinates.

This benchmark labels generic visible human figures only; it does not encode team/enemy identity and is not a live gameplay integration.
