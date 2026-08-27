import CoreFoundation
import CoreMedia
import Foundation
import ReplayKit

private enum BroadcastSignalName {
    static let started = "com.phoenix.realtimevisionassist.broadcast.started" as CFString
    static let heartbeat = "com.phoenix.realtimevisionassist.broadcast.heartbeat" as CFString
    static let finished = "com.phoenix.realtimevisionassist.broadcast.finished" as CFString
}

/// Compatibility-first ReplayKit upload extension.
///
/// No Vision/Core ML/App Group. It only keeps ReplayKit alive and emits
/// entitlement-free Darwin notifications so the main app can verify that
/// this specific Broadcast Extension is actually running.
final class BroadcastSampleHandler: RPBroadcastSampleHandler {
    private var lastHeartbeatUptime: TimeInterval = 0

    override func broadcastStarted(withSetupInfo setupInfo: [String : NSObject]?) {
        lastHeartbeatUptime = 0
        post(BroadcastSignalName.started)
        post(BroadcastSignalName.heartbeat)
    }

    override func broadcastPaused() {}

    override func broadcastResumed() {
        post(BroadcastSignalName.heartbeat)
    }

    override func broadcastFinished() {
        post(BroadcastSignalName.finished)
    }

    override func processSampleBuffer(
        _ sampleBuffer: CMSampleBuffer,
        with sampleBufferType: RPSampleBufferType
    ) {
        switch sampleBufferType {
        case .video, .audioApp, .audioMic:
            let now = ProcessInfo.processInfo.systemUptime
            if now - lastHeartbeatUptime >= 0.75 {
                lastHeartbeatUptime = now
                post(BroadcastSignalName.heartbeat)
            }
        @unknown default:
            break
        }
    }

    private func post(_ name: CFString) {
        CFNotificationCenterPostNotification(
            CFNotificationCenterGetDarwinNotifyCenter(),
            CFNotificationName(rawValue: name),
            nil,
            nil,
            true
        )
    }
}
