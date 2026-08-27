import CoreMedia
import Foundation
import ReplayKit

/// Compatibility-first ReplayKit upload extension.
///
/// This target intentionally performs no Vision/Core ML work and uses no App Group.
/// It exists to prove that the signed Broadcast Upload Extension can stay alive on-device.
/// Once this baseline is stable, analysis can be reintroduced incrementally.
final class BroadcastSampleHandler: RPBroadcastSampleHandler {
    override func broadcastStarted(withSetupInfo setupInfo: [String : NSObject]?) {
        // Deliberately empty: avoid startup allocations and entitlement-dependent I/O.
    }

    override func broadcastPaused() {}

    override func broadcastResumed() {}

    override func broadcastFinished() {}

    override func processSampleBuffer(
        _ sampleBuffer: CMSampleBuffer,
        with sampleBufferType: RPSampleBufferType
    ) {
        // Keep the ReplayKit pipeline alive while discarding samples in RAM.
        // No recording, screenshots, files, network traffic, Vision or Core ML.
        switch sampleBufferType {
        case .video, .audioApp, .audioMic:
            break
        @unknown default:
            break
        }
    }
}
