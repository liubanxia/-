import SwiftUI

final class OverlayState: ObservableObject {
    @Published var targets: [RealtimeTarget] = []
    @Published var soundIndicators: [SoundIndicatorObservation] = []
}

struct DotOverlayView: View {
    @ObservedObject var state: OverlayState

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color.clear

                ForEach(state.targets) { target in
                    ForEach(Array(target.predictedPoints.enumerated()), id: \.offset) { index, predicted in
                        Circle()
                            .fill(Color.blue.opacity(0.72 - Double(index) * 0.16))
                            .frame(width: 3, height: 3)
                            .position(
                                x: geometry.size.width * predicted.x,
                                y: geometry.size.height * predicted.y
                            )
                    }

                    if target.isVisible {
                        Circle()
                            .fill(Color.red.opacity(0.35 + 0.65 * target.audioProximity))
                            .frame(width: 4, height: 4)
                            .position(
                                x: geometry.size.width * target.point.x,
                                y: geometry.size.height * target.point.y
                            )
                    }
                }

                ForEach(Array(state.soundIndicators.prefix(2).enumerated()), id: \.offset) { _, indicator in
                    soundCue(indicator, size: geometry.size)
                }
            }
            .allowsHitTesting(false)
        }
    }

    private func soundCue(_ indicator: SoundIndicatorObservation, size: CGSize) -> some View {
        let normalizedX = min(0.96, max(0.04, 0.5 + indicator.horizontal * 0.46))
        let diameter = soundCueDiameter(indicator.distance)
        let opacity = min(0.92, max(0.35, indicator.confidence))
        let cueColor: Color = indicator.kind == .gunfire ? .red : .white

        return Circle()
            .fill(cueColor.opacity(opacity))
            .frame(width: diameter, height: diameter)
            .overlay(
                Circle().stroke(Color.black.opacity(0.42), lineWidth: 0.5)
            )
            .position(
                x: size.width * normalizedX,
                y: max(8, size.height * 0.055)
            )
    }

    private func soundCueDiameter(_ distance: SoundIndicatorObservation.DistanceBand) -> CGFloat {
        switch distance {
        case .near: return 5
        case .medium: return 4
        case .far: return 3
        }
    }
}
