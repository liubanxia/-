import CoreMedia

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

    func reset() {
        detector.reset()
        audioAnalyzer.reset()
    }
}
