// Not in the App Store build (the APPSTORE condition of the "TurboMLX App Store" target), which
// does not link Sparkle: the App Store updates the app itself.
#if !APPSTORE
import Foundation
import Observation
import Sparkle

/// Updates Turbo MLX from its GitHub releases with Sparkle. The feed is the `appcast.xml` attached
/// to the latest release (`SUFeedURL` in TurboMLX-Info.plist); an update is installed only when
/// its EdDSA signature matches `SUPublicEDKey` and it is signed by the same Developer ID, and
/// Sparkle's installer service, outside the sandbox, replaces the app and relaunches it. Checks run
/// once a day while they are on; scripts/release.sh writes and signs each release's feed.
@Observable
final class AppUpdater {
    @ObservationIgnored private let controller: SPUStandardUpdaterController
    @ObservationIgnored private var observations: [NSKeyValueObservation] = []

    /// False while a check or an update is under way.
    private(set) var canCheckForUpdates = false
    /// Sparkle's settings, mirrored here: its own windows change them too.
    private(set) var checksAutomatically = true
    private(set) var installsAutomatically = false

    init() {
        controller = SPUStandardUpdaterController(startingUpdater: true, updaterDelegate: nil, userDriverDelegate: nil)
        let updater = controller.updater
        let changed: @Sendable () -> Void = { [weak self] in Task { @MainActor in self?.refresh() } }
        observations = [
            updater.observe(\.canCheckForUpdates) { _, _ in changed() },
            updater.observe(\.automaticallyChecksForUpdates) { _, _ in changed() },
            updater.observe(\.automaticallyDownloadsUpdates) { _, _ in changed() },
        ]
        refresh()
    }

    /// Turbo MLX → Check for Updates, and Check Now: Sparkle says what it found, "up to date" too.
    func checkForUpdates() {
        controller.checkForUpdates(nil)
    }

    func setChecksAutomatically(_ on: Bool) {
        controller.updater.automaticallyChecksForUpdates = on
        refresh()
    }

    /// With it, an update is downloaded and installed when the app quits, without asking.
    func setInstallsAutomatically(_ on: Bool) {
        controller.updater.automaticallyDownloadsUpdates = on
        refresh()
    }

    func refresh() {
        let updater = controller.updater
        canCheckForUpdates = updater.canCheckForUpdates
        checksAutomatically = updater.automaticallyChecksForUpdates
        installsAutomatically = updater.automaticallyDownloadsUpdates
    }
}
#endif
