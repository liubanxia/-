import Foundation
import ReplayKit

/// ReplayKit entry point for screen-visible evidence only.
///
/// Video evidence is split into four paths:
/// - multiscale people detection/tracking for small mobile-game characters,
/// - mobile soundwave/arrow HUD recognition,
/// - low-frequency top-compass OCR for automatic heading,
/// - low-frequency visible map/POI OCR for automatic map and coarse position locking.
///
/// App audio keeps aggregate diagnostics plus a separate HRTF-oriented stereo delay/coherence
/// path. Microphone audio is intentionally ignored.
final class EnhancedBroadcastSampleHandler: RPBroadcastSampleHandler {
    // Keep the legacy processor alive only for lifecycle/heartbeat state. Build 37 no longer sends
    // video frames through its full-frame-only person path, because that path shrinks small distant
    // game characters too aggressively. Visible person evidence comes from multiscalePersonAnalyzer.
    private let visionLifecycle = EvidenceFirstVideoProcessor()
    private let multiscalePersonAnalyzer = MultiScalePersonAnalyzer()
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
        multiscalePersonAnalyzer.reset()
        visionLifecycle.start()
    }

    override func broadcastPaused() {
        multiscalePersonAnalyzer.pause()
        visionLifecycle.pause()
    }

    override func broadcastResumed() {
        multiscalePersonAnalyzer.resume()
        visionLifecycle.resume()
    }

    override func broadcastFinished() {
        mapLocalizationAnalyzer.finish()
        compassAnalyzer.finish()
        hudSoundAnalyzer.finish()
        multiscalePersonAnalyzer.finish()
        spatialAudio.finish()
        audioDiagnostics.finish()
        visionLifecycle.finish()
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
            multiscalePersonAnalyzer.consumeVideo(sampleBuffer)
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
