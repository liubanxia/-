import SwiftUI

final class OverlayState: ObservableObject {
    @Published var targets: [RealtimeTarget] = []
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
            }
            .allowsHitTesting(false)
        }
    }
}
