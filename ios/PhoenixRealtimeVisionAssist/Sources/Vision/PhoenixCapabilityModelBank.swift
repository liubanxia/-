import Foundation

/// Phoenix-style split-capability registry for LiteView's offline vision bank.
///
/// The bank can be large on disk, but this registry never loads a model. Runtime components ask
/// for one capability at a time and may keep only the selected model resident. This mirrors the
/// Phoenix approach of separating localization, segmentation, representation, scene routing and
/// depth/geometry abilities instead of treating one large model as the only source of truth.
enum PhoenixVisionCapability: String, Sendable, CaseIterable {
    case visibleLocalization
    case segmentation
    case sceneRouting
    case featureEmbedding
    case depthGeometry
}

enum PhoenixModelResidency: String, Sendable {
    case hotNano
    case warmFallback
    case coldOnly
}

struct PhoenixCapabilityModelDescriptor: Sendable, Hashable {
    let resourceName: String
    let capability: PhoenixVisionCapability
    let residency: PhoenixModelResidency
    let priority: Int
}

enum PhoenixCapabilityModelBank {
    static let descriptors: [PhoenixCapabilityModelDescriptor] = [
        // Visible-content localization. Tiny/quantized models are tried before heavier fallbacks.
        .init(resourceName: "yolo11n", capability: .visibleLocalization, residency: .hotNano, priority: 0),
        .init(resourceName: "YOLOv3TinyInt8LUT", capability: .visibleLocalization, residency: .hotNano, priority: 1),
        .init(resourceName: "YOLOv3TinyFP16", capability: .visibleLocalization, residency: .warmFallback, priority: 2),
        .init(resourceName: "YOLOv3Tiny", capability: .visibleLocalization, residency: .warmFallback, priority: 3),
        .init(resourceName: "YOLOv3Int8LUT", capability: .visibleLocalization, residency: .warmFallback, priority: 4),
        .init(resourceName: "YOLOv3FP16", capability: .visibleLocalization, residency: .coldOnly, priority: 5),
        .init(resourceName: "YOLOv3", capability: .visibleLocalization, residency: .coldOnly, priority: 6),

        // Phoenix-style separated abilities. These remain cold unless a future lane explicitly
        // requests that capability; they are never walked by the person-detector failover loop.
        .init(resourceName: "DeepLabV3Int8LUT", capability: .segmentation, residency: .coldOnly, priority: 0),
        .init(resourceName: "DeepLabV3FP16", capability: .segmentation, residency: .coldOnly, priority: 1),
        .init(resourceName: "DeepLabV3", capability: .segmentation, residency: .coldOnly, priority: 2),

        .init(resourceName: "MobileNetV2Int8LUT", capability: .sceneRouting, residency: .coldOnly, priority: 0),
        .init(resourceName: "MobileNetV2FP16", capability: .sceneRouting, residency: .coldOnly, priority: 1),
        .init(resourceName: "MobileNetV2", capability: .sceneRouting, residency: .coldOnly, priority: 2),
        .init(resourceName: "Resnet50Int8LUT", capability: .sceneRouting, residency: .coldOnly, priority: 3),
        .init(resourceName: "Resnet50FP16", capability: .sceneRouting, residency: .coldOnly, priority: 4),
        .init(resourceName: "Resnet50", capability: .sceneRouting, residency: .coldOnly, priority: 5),

        .init(resourceName: "Resnet50Headless", capability: .featureEmbedding, residency: .coldOnly, priority: 0),
        .init(resourceName: "FastViTT8F16Headless", capability: .featureEmbedding, residency: .coldOnly, priority: 1),

        .init(resourceName: "DepthAnythingV2SmallF16P6", capability: .depthGeometry, residency: .coldOnly, priority: 0),
        .init(resourceName: "DepthAnythingV2SmallF16", capability: .depthGeometry, residency: .coldOnly, priority: 1)
    ]

    static func modelURLs(for capability: PhoenixVisionCapability) -> [URL] {
        descriptors
            .filter { $0.capability == capability }
            .sorted { lhs, rhs in
                if lhs.priority != rhs.priority { return lhs.priority < rhs.priority }
                return lhs.resourceName < rhs.resourceName
            }
            .compactMap { Bundle.main.url(forResource: $0.resourceName, withExtension: "mlmodelc") }
    }

    static var visibleLocalizationURLs: [URL] {
        modelURLs(for: .visibleLocalization)
    }

    static func installedDescriptors() -> [PhoenixCapabilityModelDescriptor] {
        descriptors.filter {
            Bundle.main.url(forResource: $0.resourceName, withExtension: "mlmodelc") != nil
        }
    }
}
