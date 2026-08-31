import Foundation
import ReplayKit

/// Build 32 ReplayKit entry point.
///
/// Video is consumed by two independent screen-visible paths:
/// - EvidenceFirstVideoProcessor: people detection/tracking.
/// - BroadcastHUDSoundAnalyzer: rendered mobile soundwave/arrow HUD analysis.
///
/// App audio keeps aggregate telemetry for diagnostics and a separate HRTF-oriented stereo
/// delay/coherence path. Microphone audio is intentionally ignored.
final class EnhancedBroadcastSampleHandler: RPBroadcastSampleHandler {
    private let visionProcessor = EvidenceFirstVideoProcessor()
    private let hudSoundAnalyzer = BroadcastHUDSoundAnalyzer()
    private let audioDiagnostics = BroadcastAudioTelemetryAnalyzer()
    private let spatialAudio = BroadcastSpatialAudioAnalyzer()

    override func broadcastStarted(withSetupInfo setupInfo: [String: NSObject]?) {
        audioDiagnostics.reset()
        spatialAudio.reset()
        hudSoundAnalyzer.reset()
        visionProcessor.start()
    }

    override func broadcastPaused() {
        visionProcessor.pause()
    }

    override func broadcastResumed() {
        visionProcessor.resume()
    }

    override func broadcastFinished() {
        hudSoundAnalyzer.finish()
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
            hudSoundAnalyzer.consumeVideo(sampleBuffer)
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
