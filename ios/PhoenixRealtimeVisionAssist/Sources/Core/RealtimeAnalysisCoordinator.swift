import CoreMedia
import Foundation

final class RealtimeAnalysisCoordinator {
    private let detector: RealtimePersonDetector
    private let audioAnalyzer: RealtimeAudioAnalyzer
    private let soundIndicatorAnalyzer: SoundIndicatorROIAnalyzer
    private let soundIndicatorStabilizer: SoundIndicatorStabilizer

    private(set) var soundIndicators: [SoundIndicatorObservation] = []

    init(configuration: RuntimeConfiguration = .default) {
        self.detector = RealtimePersonDetector(configuration: configuration)
        self.audioAnalyzer = RealtimeAudioAnalyzer()
        self.soundIndicatorAnalyzer = SoundIndicatorROIAnalyzer()
        self.soundIndicatorStabilizer = SoundIndicatorStabilizer()
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer, onTargets: @escaping @Sendable ([RealtimeTarget]) -> Void) {
        let rawIndicators = soundIndicatorAnalyzer.analyze(sampleBuffer)
        soundIndicators = soundIndicatorStabilizer.update(rawIndicators)

        detector.analyze(
            sampleBuffer: sampleBuffer,
            audioProximity: audioAnalyzer.proximity,
            completion: onTargets
        )
    }

    func consumeAudio(_ sampleBuffer: CMSampleBuffer) {
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
        detector.reset()
        audioAnalyzer.reset()
        soundIndicatorAnalyzer.reset()
        soundIndicatorStabilizer.reset()
        soundIndicators.removeAll(keepingCapacity: false)
    }
}
