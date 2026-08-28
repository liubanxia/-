import Foundation

@main
enum LiteViewLifecycleSmokeMain {
    static func main() {
        var state = BroadcastLifecycleState()
        precondition(state.phase == .ready)
        precondition(!state.apply(.started, now: 1))
        precondition(state.phase == .running)
        precondition(state.apply(.finished, now: 2))
        precondition(state.phase == .recovering)
        precondition(!state.apply(.pickerRebuilt, now: 3))
        precondition(state.phase == .ready)
        precondition(!state.apply(.started, now: 4))
        precondition(state.phase == .running)

        let framesOnly = SharedRealtimeSnapshot(
            sessionID: "smoke",
            sequence: 1,
            phase: .running,
            targetCount: 0,
            soundIndicatorCount: 0,
            videoFrameCount: 5,
            videoFramesPerSecond: 30,
            droppedAnalysisFrameCount: 0,
            analysisLatencyMilliseconds: 0,
            analysisMode: .lightweightVision
        )
        precondition(framesOnly.visionPipelineStage == .framesReceived)

        let stableTarget = SharedRealtimeSnapshot(
            sessionID: "smoke",
            sequence: 2,
            phase: .running,
            timestamp: 100,
            targetCount: 1,
            soundIndicatorCount: 0,
            videoFrameCount: 10,
            videoFramesPerSecond: 30,
            droppedAnalysisFrameCount: 0,
            analysisLatencyMilliseconds: 20,
            analysisMode: .lightweightVision,
            analysisFrameCount: 3,
            successfulAnalysisFrameCount: 3,
            lastAnalysisSucceeded: true,
            attemptedLaneCount: 2,
            successfulLaneCount: 2,
            primaryTarget: SharedNormalizedPoint(x: 0.4, y: 0.6),
            primaryTargetConfidence: 0.9,
            stableTargetFrameCount: 3
        )
        precondition(stableTarget.visionPipelineStage == .stableTarget)

        let compact = CompactBroadcastState(snapshot: stableTarget)
        let restored = CompactBroadcastState(rawValue: compact.rawValue)?.makeSnapshot(at: 100.25)
        precondition(restored?.visionPipelineStage == .stableTarget)
        precondition(abs((restored?.primaryTarget?.x ?? -1) - 0.4) < 0.01)
        precondition(abs((restored?.primaryTarget?.y ?? -1) - 0.6) < 0.01)

        let inferenceFailure = SharedRealtimeSnapshot(
            sessionID: "smoke-failure",
            sequence: 3,
            phase: .running,
            timestamp: 200,
            targetCount: 0,
            soundIndicatorCount: 0,
            videoFrameCount: 10,
            videoFramesPerSecond: 30,
            droppedAnalysisFrameCount: 0,
            analysisLatencyMilliseconds: 20,
            analysisMode: .lightweightVision,
            analysisFrameCount: 1,
            successfulAnalysisFrameCount: 0,
            lastAnalysisSucceeded: false,
            attemptedLaneCount: 1,
            successfulLaneCount: 0
        )
        let failedCompact = CompactBroadcastState(snapshot: inferenceFailure)
        let failedRestored = CompactBroadcastState(rawValue: failedCompact.rawValue)?.makeSnapshot(at: 200.25)
        precondition(failedRestored?.visionPipelineStage == .inferenceFailed)

        let realtimeModels = PhoenixCapabilityModelBank.descriptors(for: .visibleLocalization)
        precondition(!realtimeModels.isEmpty)
        precondition(realtimeModels.allSatisfy { $0.residency != .coldOnly })
        precondition(realtimeModels.allSatisfy { $0.capability == .visibleLocalization })

        print("PASS: LiteView lifecycle, compact coordinate fallback, AI evidence, and residency policy")
    }
}
