import ReplayKit

final class BroadcastSampleHandler: RPBroadcastSampleHandler {
    private let coordinator = RealtimeAnalysisCoordinator()
    private let sharedState = SharedRealtimeStateStore()
    private var latestTargets: [RealtimeTarget] = []

    override func broadcastStarted(withSetupInfo setupInfo: [String : NSObject]?) {
        coordinator.reset()
        latestTargets.removeAll(keepingCapacity: false)
        sharedState.clear()
    }

    override func broadcastPaused() {}

    override func broadcastResumed() {}

    override func broadcastFinished() {
        coordinator.reset()
        latestTargets.removeAll(keepingCapacity: false)
        sharedState.clear()
    }

    override func processSampleBuffer(_ sampleBuffer: CMSampleBuffer, with sampleBufferType: RPSampleBufferType) {
        switch sampleBufferType {
        case .video:
            coordinator.consumeVideo(sampleBuffer) { [weak self] targets in
                guard let self else { return }
                self.latestTargets = targets
                self.sharedState.publish(
                    targetCount: targets.count,
                    soundIndicatorCount: self.coordinator.soundIndicators.count
                )
            }
        case .audioApp:
            coordinator.consumeAudio(sampleBuffer)
        case .audioMic:
            break
        @unknown default:
            break
        }
    }
}
