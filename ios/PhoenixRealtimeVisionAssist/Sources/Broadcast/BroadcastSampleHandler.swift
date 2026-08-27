import CoreMedia
import Foundation
import ReplayKit
import Vision

final class BroadcastSampleHandler: RPBroadcastSampleHandler {
    private let sharedState = SharedRealtimeStateStore()
    private let analysisQueue = DispatchQueue(label: "phoenix.broadcast.visible-human", qos: .utility)
    private let stateLock = NSLock()

    private var lastAnalysisTime: TimeInterval = 0
    private var analysisInFlight = false
    private let analysisInterval: TimeInterval = 0.5

    override func broadcastStarted(withSetupInfo setupInfo: [String : NSObject]?) {
        stateLock.lock()
        lastAnalysisTime = 0
        analysisInFlight = false
        stateLock.unlock()

        sharedState.clear()
        sharedState.publish(targetCount: 0, soundIndicatorCount: 0)
    }

    override func broadcastPaused() {}

    override func broadcastResumed() {}

    override func broadcastFinished() {
        stateLock.lock()
        analysisInFlight = false
        stateLock.unlock()
        sharedState.clear()
    }

    override func processSampleBuffer(
        _ sampleBuffer: CMSampleBuffer,
        with sampleBufferType: RPSampleBufferType
    ) {
        guard sampleBufferType == .video,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let now = ProcessInfo.processInfo.systemUptime
        guard reserveAnalysis(at: now) else { return }

        analysisQueue.async { [weak self] in
            guard let self else { return }
            defer { self.finishAnalysis() }

            let visibleCount = self.visibleHumanCount(in: pixelBuffer)
            self.sharedState.publish(
                targetCount: visibleCount,
                soundIndicatorCount: 0
            )
        }
    }

    private func reserveAnalysis(at now: TimeInterval) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }

        guard !analysisInFlight else { return false }
        guard now - lastAnalysisTime >= analysisInterval else { return false }

        analysisInFlight = true
        lastAnalysisTime = now
        return true
    }

    private func finishAnalysis() {
        stateLock.lock()
        analysisInFlight = false
        stateLock.unlock()
    }

    private func visibleHumanCount(in pixelBuffer: CVPixelBuffer) -> Int {
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        do {
            try handler.perform([request])
            return (request.results ?? []).filter { $0.confidence >= 0.35 }.count
        } catch {
            return 0
        }
    }
}
