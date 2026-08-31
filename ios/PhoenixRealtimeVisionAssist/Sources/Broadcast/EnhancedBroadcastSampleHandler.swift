import Foundation
import ReplayKit

/// Build 32 ReplayKit entry point.
///
/// Screen-visible video evidence is split into three paths:
/// - people detection/tracking,
/// - mobile soundwave/arrow HUD recognition,
/// - low-frequency top-compass OCR for automatic map heading.
///
/// App audio keeps aggregate diagnostics plus a separate HRTF-oriented stereo delay/coherence
/// path. Microphone audio is intentionally ignored.
final class EnhancedBroadcastSampleHandler: RPBroadcastSampleHandler {
    private let visionProcessor = EvidenceFirstVideoProcessor()
    private let hudSoundAnalyzer = BroadcastHUDSoundAnalyzer()
    private let compassAnalyzer = BroadcastCompassHeadingAnalyzer()
    private let audioDiagnostics = BroadcastAudioTelemetryAnalyzer()
    private let spatialAudio = BroadcastSpatialAudioAnalyzer()

    override func broadcastStarted(withSetupInfo setupInfo: [String: NSObject]?) {
        audioDiagnostics.reset()
        spatialAudio.reset()
        hudSoundAnalyzer.reset()
        compassAnalyzer.reset()
        visionProcessor.start()
    }

    override func broadcastPaused() {
        visionProcessor.pause()
    }

    override func broadcastResumed() {
        visionProcessor.resume()
    }

    override func broadcastFinished() {
        compassAnalyzer.finish()
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
            compassAnalyzer.consumeVideo(sampleBuffer)
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
