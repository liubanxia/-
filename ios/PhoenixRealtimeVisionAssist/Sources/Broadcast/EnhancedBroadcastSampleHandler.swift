import Foundation
import ReplayKit

/// ReplayKit entry point for screen-visible evidence only.
///
/// Video evidence is split into four paths:
/// - people detection/tracking,
/// - mobile soundwave/arrow HUD recognition,
/// - low-frequency top-compass OCR for automatic heading,
/// - low-frequency visible map/POI OCR for automatic map and coarse position locking.
///
/// App audio keeps aggregate diagnostics plus a separate HRTF-oriented stereo delay/coherence
/// path. Microphone audio is intentionally ignored.
final class EnhancedBroadcastSampleHandler: RPBroadcastSampleHandler {
    private let visionProcessor = EvidenceFirstVideoProcessor()
    private let hudSoundAnalyzer = BroadcastHUDSoundAnalyzer()
    private let compassAnalyzer = BroadcastCompassHeadingAnalyzer()
    private let mapLocalizationAnalyzer = BroadcastMapLocalizationAnalyzer()
    private let audioDiagnostics = BroadcastAudioTelemetryAnalyzer()
    private let spatialAudio = BroadcastSpatialAudioAnalyzer()

    override func broadcastStarted(withSetupInfo setupInfo: [String: NSObject]?) {
        audioDiagnostics.reset()
        spatialAudio.reset()
        hudSoundAnalyzer.reset()
        compassAnalyzer.reset()
        mapLocalizationAnalyzer.reset()
        visionProcessor.start()
    }

    override func broadcastPaused() {
        visionProcessor.pause()
    }

    override func broadcastResumed() {
        visionProcessor.resume()
    }

    override func broadcastFinished() {
        mapLocalizationAnalyzer.finish()
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
            mapLocalizationAnalyzer.consumeVideo(sampleBuffer)
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
