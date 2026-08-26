# LiteView model staging

The realtime iOS target keeps model weights out of Git.

Current optional generic detector cache:

- Source: YOLO11n `.pt`
- Conversion: GitHub Actions macOS workflow `.github/workflows/liteview-coreml-export.yml`
- Output artifact: `liteview-yolo11n-coreml`
- Runtime integration target: Core ML / Vision

The Apple Vision baseline remains available when no external Core ML detector is present.
