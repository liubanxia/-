import Foundation
import ReplayKit

/// ReplayKit entry point that preserves the validated video pipeline and adds real `.audioApp`
/// diagnostics without changing the stable vision analyzer implementation.
final class EnhancedBroadcastSampleHandler: RPBroadcastSampleHandler {
    private let visionHandler = BroadcastSampleHandler()
    private let audioAnalyzer = BroadcastAudioTelemetryAnalyzer()

    override func broadcastStarted(withSetupInfo setupInfo: [String: NSObject]?) {
        audioAnalyzer.reset()
        visionHandler.broadcastStarted(withSetupInfo: setupInfo)
    }

    override func broadcastPaused() {
        visionHandler.broadcastPaused()
    }

    override func broadcastResumed() {
        visionHandler.broadcastResumed()
    }

    override func broadcastFinished() {
        audioAnalyzer.finish()
        visionHandler.broadcastFinished()
    }

    override func processSampleBuffer(
        _ sampleBuffer: CMSampleBuffer,
        with sampleBufferType: RPSampleBufferType
    ) {
        switch sampleBufferType {
        case .video:
            visionHandler.processSampleBuffer(sampleBuffer, with: .video)
        case .audioApp:
            audioAnalyzer.consume(sampleBuffer)
        case .audioMic:
            // Microphone audio is intentionally ignored. LiteView only diagnoses the app/game
            // audio stream and never records or persists microphone content.
            break
        @unknown default:
            break
        }
    }
}
