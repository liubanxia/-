import Foundation

/// Pixel-centric capability registry for LiteView's offline vision bank.
///
/// The bank is allowed to be large on disk, but every model must provide a distinct visual role.
/// Duplicate classifier/precision variants that do not improve localization, segmentation,
/// representation or geometry are intentionally excluded.
enum PhoenixVisionCapability: String, Sendable, CaseIterable {
    case visibleLocalization
    case segmentation
    case featureEmbedding
    case depthGeometry
}

enum PhoenixModelResidency: String, Sendable, CaseIterable, Hashable {
    case hotNano
    case warmFallback
    case coldOnly
}

enum PhoenixPixelRole: String, Sendable, CaseIterable, Hashable {
    case fastLocalization
    case highRecallLocalization
    case pixelSegmentation
    case appearanceEmbedding
    case depthGeometry
}

struct PhoenixCapabilityModelDescriptor: Sendable, Hashable {
    let resourceName: String
    let capability: PhoenixVisionCapability
    let pixelRole: PhoenixPixelRole
    let residency: PhoenixModelResidency
    let priority: Int
    let preferredInputEdge: Int
}

enum PhoenixCapabilityModelBank {
    /// Only the two tiny localization probes are eligible for lightweight runtime experiments.
    /// The rebuilt large bank stays cold until an offline benchmark explicitly requests it.
    static let realtimeResidencies: Set<PhoenixModelResidency> = [.hotNano]

    static let descriptors: [PhoenixCapabilityModelDescriptor] = [
        // Localization: retain architecturally/precision-distinct probes, not every duplicate.
        .init(
            resourceName: "yolo11n",
            capability: .visibleLocalization,
            pixelRole: .fastLocalization,
            residency: .hotNano,
            priority: 0,
            preferredInputEdge: 640
        ),
        .init(
            resourceName: "YOLOv3TinyInt8LUT",
            capability: .visibleLocalization,
            pixelRole: .fastLocalization,
            residency: .hotNano,
            priority: 1,
            preferredInputEdge: 416
        ),
        .init(
            resourceName: "YOLOv3Int8LUT",
            capability: .visibleLocalization,
            pixelRole: .highRecallLocalization,
            residency: .coldOnly,
            priority: 2,
            preferredInputEdge: 416
        ),
        .init(
            resourceName: "YOLOv3FP16",
            capability: .visibleLocalization,
            pixelRole: .highRecallLocalization,
            residency: .coldOnly,
            priority: 3,
            preferredInputEdge: 416
        ),

        // Pixel segmentation: keep a fast quantized path and a higher-precision comparison path.
        .init(
            resourceName: "DeepLabV3Int8LUT",
            capability: .segmentation,
            pixelRole: .pixelSegmentation,
            residency: .coldOnly,
            priority: 0,
            preferredInputEdge: 513
        ),
        .init(
            resourceName: "DeepLabV3FP16",
            capability: .segmentation,
            pixelRole: .pixelSegmentation,
            residency: .coldOnly,
            priority: 1,
            preferredInputEdge: 513
        ),

        // Feature representation: headless networks keep the useful visual embedding while
        // dropping redundant ImageNet classifier heads.
        .init(
            resourceName: "Resnet50Headless",
            capability: .featureEmbedding,
            pixelRole: .appearanceEmbedding,
            residency: .coldOnly,
            priority: 0,
            preferredInputEdge: 224
        ),
        .init(
            resourceName: "FastViTT8F16Headless",
            capability: .featureEmbedding,
            pixelRole: .appearanceEmbedding,
            residency: .coldOnly,
            priority: 1,
            preferredInputEdge: 256
        ),
        .init(
            resourceName: "FastViTMA36F16Headless",
            capability: .featureEmbedding,
            pixelRole: .appearanceEmbedding,
            residency: .coldOnly,
            priority: 2,
            preferredInputEdge: 256
        ),

        // Geometry reserve. Two published variants remain because they are not simple classifier
        // duplicates and can be compared for relative-depth consistency on the offline benchmark.
        .init(
            resourceName: "DepthAnythingV2SmallF16P6",
            capability: .depthGeometry,
            pixelRole: .depthGeometry,
            residency: .coldOnly,
            priority: 0,
            preferredInputEdge: 518
        ),
        .init(
            resourceName: "DepthAnythingV2SmallF16",
            capability: .depthGeometry,
            pixelRole: .depthGeometry,
            residency: .coldOnly,
            priority: 1,
            preferredInputEdge: 518
        )
    ]

    static func descriptors(
        for capability: PhoenixVisionCapability,
        allowedResidencies: Set<PhoenixModelResidency> = realtimeResidencies
    ) -> [PhoenixCapabilityModelDescriptor] {
        descriptors
            .filter {
                $0.capability == capability && allowedResidencies.contains($0.residency)
            }
            .sorted { lhs, rhs in
                if lhs.priority != rhs.priority { return lhs.priority < rhs.priority }
                return lhs.resourceName < rhs.resourceName
            }
    }

    static func offlineDescriptors(for capability: PhoenixVisionCapability) -> [PhoenixCapabilityModelDescriptor] {
        descriptors
            .filter { $0.capability == capability }
            .sorted { lhs, rhs in
                if lhs.priority != rhs.priority { return lhs.priority < rhs.priority }
                return lhs.resourceName < rhs.resourceName
            }
    }

    static func modelURLs(
        for capability: PhoenixVisionCapability,
        allowedResidencies: Set<PhoenixModelResidency> = realtimeResidencies,
        bundle: Bundle = .main
    ) -> [URL] {
        descriptors(for: capability, allowedResidencies: allowedResidencies)
            .compactMap { bundle.url(forResource: $0.resourceName, withExtension: "mlmodelc") }
    }

    static var visibleLocalizationURLs: [URL] {
        modelURLs(for: .visibleLocalization)
    }

    static func installedDescriptors(bundle: Bundle = .main) -> [PhoenixCapabilityModelDescriptor] {
        descriptors.filter {
            bundle.url(forResource: $0.resourceName, withExtension: "mlmodelc") != nil
        }
    }
}
