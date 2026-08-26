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
                    Circle()
                        .fill(Color.red.opacity(0.35 + 0.65 * target.audioProximity))
                        .frame(width: 8, height: 8)
                        .position(
                            x: geometry.size.width * target.point.x,
                            y: geometry.size.height * target.point.y
                        )
                }
            }
            .allowsHitTesting(false)
        }
    }
}
