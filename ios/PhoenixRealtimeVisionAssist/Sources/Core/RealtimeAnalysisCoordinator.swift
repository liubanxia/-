import CoreMedia
import Foundation

final class RealtimeAnalysisCoordinator {
    private let configuration: RuntimeConfiguration
    private let detector: RealtimePersonDetector
    private let audioAnalyzer: RealtimeAudioAnalyzer
    private let soundIndicatorAnalyzer: SoundIndicatorROIAnalyzer
    private let soundIndicatorStabilizer: SoundIndicatorStabilizer

    private(set) var soundIndicators: [SoundIndicatorObservation] = []

    init(configuration: RuntimeConfiguration = .default) {
        self.configuration = configuration
        self.detector = RealtimePersonDetector(configuration: configuration)
        self.audioAnalyzer = RealtimeAudioAnalyzer()
        self.soundIndicatorAnalyzer = SoundIndicatorROIAnalyzer()
        self.soundIndicatorStabilizer = SoundIndicatorStabilizer()
    }

    func consumeVideo(
        _ sampleBuffer: CMSampleBuffer,
        onTargets: @escaping @Sendable ([RealtimeTarget]) -> Void
    ) {
        if configuration.enableScreenCueAnalysis {
            let rawIndicators = soundIndicatorAnalyzer.analyze(sampleBuffer)
            soundIndicators = soundIndicatorStabilizer.update(rawIndicators)
        } else if !soundIndicators.isEmpty {
            soundIndicators.removeAll(keepingCapacity: false)
        }

        detector.analyze(
            sampleBuffer: sampleBuffer,
            audioProximity: configuration.enableAudioLevelAnalysis ? audioAnalyzer.proximity : 0,
            completion: onTargets
        )
    }

    func consumeAudio(_ sampleBuffer: CMSampleBuffer) {
        guard configuration.enableAudioLevelAnalysis else { return }
        audioAnalyzer.consume(sampleBuffer: sampleBuffer)
    }

    func updateMapContext(_ context: MapPredictionContext?) {
        detector.updateMapContext(context)
    }

    func replaceMapKnowledge(_ knowledge: MapKnowledge) {
        detector.replaceMapKnowledge(knowledge)
    }

    func loadMapKnowledgeJSON(_ data: Data) throws {
        try detector.loadMapKnowledgeJSON(data)
    }

    func reset() {
        detector.releaseHeavyResources()
        audioAnalyzer.reset()
        soundIndicatorAnalyzer.reset()
        soundIndicatorStabilizer.reset()
        soundIndicators.removeAll(keepingCapacity: false)
    }
}
