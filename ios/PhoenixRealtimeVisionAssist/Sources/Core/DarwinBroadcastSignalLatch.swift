import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_latch_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_check")
private func liteview_latch_check(
    _ token: Int32,
    _ check: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_latch_cancel(_ token: Int32) -> UInt32

/// Pollable Darwin-notify latch for lifecycle events that can be posted while the main app
/// is suspended behind another foreground app. `notify_check` remembers a post until consumed,
/// unlike a live CFNotification callback which can be missed while the process is suspended.
final class DarwinBroadcastSignalLatch {
    private let lock = NSLock()
    private var tokens: [String: Int32] = [:]

    let isAvailable: Bool

    init(names: [String] = BroadcastSignalName.all) {
        var registered: [String: Int32] = [:]

        for name in names {
            var token: Int32 = -1
            let status = name.withCString {
                liteview_latch_register_check($0, &token)
            }
            guard status == 0, token >= 0 else { continue }

            // notify_check reports true on the first check. Consume that bootstrap edge now so
            // later true values represent posts that happened after this latch was armed.
            var bootstrap: Int32 = 0
            _ = liteview_latch_check(token, &bootstrap)
            registered[name] = token
        }

        tokens = registered
        isAvailable = !registered.isEmpty
    }

    deinit {
        lock.lock()
        let values = Array(tokens.values)
        tokens.removeAll(keepingCapacity: false)
        lock.unlock()

        for token in values {
            _ = liteview_latch_cancel(token)
        }
    }

    func consume() -> Set<String> {
        lock.lock()
        defer { lock.unlock() }

        var posted = Set<String>()
        for (name, token) in tokens {
            var flag: Int32 = 0
            guard liteview_latch_check(token, &flag) == 0 else { continue }
            if flag != 0 {
                posted.insert(name)
            }
        }
        return posted
    }
}
