import CoreMedia
import Foundation

final class RealtimeAnalysisCoordinator {
    private let detector: RealtimePersonDetector
    private let audioAnalyzer: RealtimeAudioAnalyzer

    init(configuration: RuntimeConfiguration = .default) {
        self.detector = RealtimePersonDetector(configuration: configuration)
        self.audioAnalyzer = RealtimeAudioAnalyzer()
    }

    func consumeVideo(_ sampleBuffer: CMSampleBuffer, onTargets: @escaping @Sendable ([RealtimeTarget]) -> Void) {
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
    }
}
