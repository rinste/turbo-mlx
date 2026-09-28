import Foundation

/// Keeps the Mac from sleeping on idle, and the app out of App Nap, while work the user started
/// runs (a download, the generation queue), so that it is not cut off halfway when they walk away.
/// The display still sleeps; closing a laptop's lid still puts it to sleep.
final class KeepAwake {
    private let reason: String
    private var activity: (any NSObjectProtocol)?

    init(reason: String) {
        self.reason = reason
    }

    /// True while something needs the Mac awake.
    var isOn: Bool {
        get { activity != nil }
        set {
            guard newValue != isOn else { return }
            if newValue {
                activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: reason)
            } else if let activity {
                ProcessInfo.processInfo.endActivity(activity)
                self.activity = nil
            }
        }
    }
}
