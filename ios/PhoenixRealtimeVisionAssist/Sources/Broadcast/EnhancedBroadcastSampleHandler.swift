import Foundation
import ReplayKit

/// Build 31 ReplayKit entry point.
/// Video uses the faster multi-target evidence processor. App audio is analyzed twice:
/// legacy aggregate telemetry remains for diagnostics while the spatial path publishes
/// inter-channel delay/coherence cues. Microphone audio is intentionally ignored.
final class EnhancedBroadcastSampleHandler: RPBroadcastSampleHandler {
    private let visionProcessor = EvidenceFirstVideoProcessor()
    private let audioDiagnostics = BroadcastAudioTelemetryAnalyzer()
    private let spatialAudio = BroadcastSpatialAudioAnalyzer()

    override func broadcastStarted(withSetupInfo setupInfo: [String: NSObject]?) {
        audioDiagnostics.reset()
        spatialAudio.reset()
        visionProcessor.start()
    }

    override func broadcastPaused() {
        visionProcessor.pause()
    }

    override func broadcastResumed() {
        visionProcessor.resume()
    }

    override func broadcastFinished() {
        spatialAudio.finish()
        audioDiagnostics.finish()
        visionProcessor.finish()
    }

    override func processSampleBuffer(
        _ sampleBuffer: CMSampleBuffer,
        with sampleBufferType: RPSampleBufferType
    ) {
        switch sampleBufferType {
        case .video:
            visionProcessor.consumeVideo(sampleBuffer)
        case .audioApp:
            audioDiagnostics.consume(sampleBuffer)
            spatialAudio.consume(sampleBuffer)
        case .audioMic:
            break
        @unknown default:
            break
        }
    }
}
