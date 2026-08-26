import CoreMedia
import Foundation

final class RealtimeAnalysisCoordinator {
    private let detector: RealtimePersonDetector
    private let audioAnalyzer: RealtimeAudioAnalyzer
    private let soundIndicatorAnalyzer: SoundIndicatorROIAnalyzer

    private(set) var soundIndicators: [SoundIndicatorObservation] = []

    init(configuration: RuntimeConfiguration = .default) {
        self.detector = RealtimePersonDetector(configuration: configuration)
        self.audioAnalyzer = RealtimeAudioAnalyzer()
        self.soundIndicatorAnalyzer = SoundIndicatorROIAnalyzer()
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer, onTargets: @escaping @Sendable ([RealtimeTarget]) -> Void) {
        soundIndicators = soundIndicatorAnalyzer.analyze(sampleBuffer)
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
        soundIndicators.removeAll(keepingCapacity: false)
    }
}
