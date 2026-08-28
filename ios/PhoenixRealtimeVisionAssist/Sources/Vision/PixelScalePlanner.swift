import CoreGraphics
import Foundation

/// Offline pixel-preserving scan planner.
///
/// The planner exists to stop small visible humans from disappearing when a full phone frame is
/// reduced to a single low-resolution detector input. It does not perform inference and it is not
/// wired to ReplayKit. Offline evaluation can use these regions to compare full-frame and tiled
/// recognition at the same source-frame pixels.
struct PixelScaleScanRegion: Sendable, Hashable {
    enum Kind: String, Sendable {
        case fullFrame
        case quadrant
        case centerBand
        case centerCrop
    }

    let kind: Kind
    /// Normalized source-image rectangle, origin at top-left.
    let rect: CGRect
    /// Suggested detector input edge. The caller may choose the nearest supported model input.
    let preferredInputEdge: Int
}

enum PixelScalePlanner {
    /// Builds a bounded multi-scale plan from the original frame dimensions.
    ///
    /// Important: this deliberately crops the original pixel buffer before resizing each region.
    /// A 30 px tall figure in a 2532x1170 frame can become ~60-100 detector pixels inside a tile
    /// instead of collapsing to only a few pixels after whole-frame downscaling.
    static func plan(
        sourceWidth: Int,
        sourceHeight: Int,
        expectedSmallTargetHeightPixels: Int = 28
    ) -> [PixelScaleScanRegion] {
        guard sourceWidth > 0, sourceHeight > 0 else { return [] }

        var regions: [PixelScaleScanRegion] = [
            .init(kind: .fullFrame, rect: CGRect(x: 0, y: 0, width: 1, height: 1), preferredInputEdge: 640)
        ]

        let shortEdge = min(sourceWidth, sourceHeight)
        let targetRatio = Double(expectedSmallTargetHeightPixels) / Double(max(shortEdge, 1))

        // When a small visible figure would be crushed by whole-frame resize, preserve source
        // pixels with overlapping 2x2 tiles. 8% overlap avoids losing a figure at tile borders.
        if targetRatio < 0.055 {
            let tile = 0.54
            let starts = [0.0, 0.46]
            for y in starts {
                for x in starts {
                    regions.append(
                        .init(
                            kind: .quadrant,
                            rect: CGRect(x: x, y: y, width: tile, height: tile),
                            preferredInputEdge: 640
                        )
                    )
                }
            }
        }

        // FPS games concentrate actionable visible figures around the middle horizontal band.
        // This is an offline visual-evaluation crop only; it does not infer team or enemy identity.
        regions.append(
            .init(
                kind: .centerBand,
                rect: CGRect(x: 0.12, y: 0.20, width: 0.76, height: 0.62),
                preferredInputEdge: 640
            )
        )

        if targetRatio < 0.035 {
            regions.append(
                .init(
                    kind: .centerCrop,
                    rect: CGRect(x: 0.24, y: 0.18, width: 0.52, height: 0.64),
                    preferredInputEdge: 768
                )
            )
        }

        return deduplicated(regions)
    }

    private static func deduplicated(_ regions: [PixelScaleScanRegion]) -> [PixelScaleScanRegion] {
        var seen: Set<String> = []
        return regions.filter { region in
            let r = region.rect
            let key = [r.origin.x, r.origin.y, r.size.width, r.size.height]
                .map { String(format: "%.4f", Double($0)) }
                .joined(separator: ":") + ":\(region.preferredInputEdge)"
            return seen.insert(key).inserted
        }
    }
}
