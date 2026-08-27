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
        print("PASS: LiteView start-stop-start lifecycle")
    }
}
