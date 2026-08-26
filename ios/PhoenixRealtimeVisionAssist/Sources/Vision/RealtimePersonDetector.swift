import CoreMedia
import Vision

final class RealtimePersonDetector {
    private let queue = DispatchQueue(label: "phoenix.vision.detector", qos: .userInitiated)
    private var lastAnalysisTime: TimeInterval = 0
    private let configuration: RuntimeConfiguration

    init(configuration: RuntimeConfiguration = .default) {
        self.configuration = configuration
    }

    func analyze(
        sampleBuffer: CMSampleBuffer,
        audioProximity: Double,
        completion: @escaping @Sendable ([RealtimeTarget]) -> Void
    ) {
        let now = ProcessInfo.processInfo.systemUptime
        let fps = currentFPS()
        guard fps > 0 else {
            completion([])
            return
        }
        guard now - lastAnalysisTime >= (1.0 / fps) else { return }
        lastAnalysisTime = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        queue.async { [configuration] in
            let request = VNDetectHumanRectanglesRequest()
            request.upperBodyOnly = false

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
            do {
                try handler.perform([request])
                let observations = (request.results ?? []).filter { $0.confidence >= configuration.minimumConfidence }
                let targets = observations.map { observation -> RealtimeTarget in
                    let box = observation.boundingBox
                    let x = box.midX
                    let y: Double
                    if configuration.useHeadBiasedPoint {
                        y = box.minY + box.height * 0.78
                    } else {
                        y = box.midY
                    }
                    return RealtimeTarget(
                        point: NormalizedPoint(x: x, y: 1.0 - y),
                        confidence: Double(observation.confidence),
                        audioProximity: audioProximity
                    )
                }
                completion(targets)
            } catch {
                completion([])
            }
        }
    }

    private func currentFPS() -> Double {
        switch ThermalBudget.current {
        case .nominal: return configuration.nominalFPS
        case .fair: return configuration.fairFPS
        case .serious: return configuration.seriousFPS
        case .critical: return 0
        }
    }
}
