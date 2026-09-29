import Foundation
import Synchronization

/// Where the Hugging Face models live. By default, outside the App Store, `~/.cache/huggingface`,
/// the folder the huggingface CLI, mflux and other tools share (the sandbox reaches it through an
/// entitlement for that one path, TurboMLX.entitlements); in the App Store build, which may not
/// have that entitlement, a folder in the app's container. Either way the user can pick another
/// folder (Settings → Models): the sandbox lets the app into it through the security-scoped
/// bookmark kept here, opened at launch before the engine starts, so that the engine, a child
/// process in the app's sandbox, inherits the access and reads the models there too.
nonisolated enum ModelFolder {
    private struct Chosen {
        var url: URL
        /// The folder is a hub cache itself (it holds `models--…` folders) rather than a Hugging
        /// Face home with a `hub` inside.
        var isHub: Bool
    }

    private static let bookmarkKey = "modelFolderBookmark"
    private static let isHubKey = "modelFolderIsHub"
    private static let pathKey = "modelFolderPath"
    private static let chosen = Mutex<Chosen?>(nil)

    /// The hub cache: `models--<org>--<name>` folders, in the Hugging Face layout.
    static var hubCache: URL {
        guard let chosen = chosen.withLock({ $0 }) else { return defaultHome.appending(path: "hub", directoryHint: .isDirectory) }
        return chosen.isHub ? chosen.url : chosen.url.appending(path: "hub", directoryHint: .isDirectory)
    }

    /// The folder the user picked and the app could open, nil while the default is used.
    static var custom: URL? { chosen.withLock { $0?.url } }

    /// A folder the user picked that could not be opened at launch (a disk not connected, a
    /// folder since deleted): the default is used meanwhile.
    static var unavailable: String? {
        guard custom == nil, UserDefaults.standard.data(forKey: bookmarkKey) != nil else { return nil }
        return UserDefaults.standard.string(forKey: pathKey)
    }

    /// The default Hugging Face home for this build.
    static var defaultHome: URL {
        #if APPSTORE
        BackendController.supportDirectory.appending(path: "Models", directoryHint: .isDirectory)
        #else
        realHome.appending(path: ".cache/huggingface", directoryHint: .isDirectory)
        #endif
    }

    /// The user's home folder: in the sandbox, FileManager's is the app's container.
    static var realHome: URL {
        if let entry = getpwuid(getuid()), let directory = entry.pointee.pw_dir {
            return URL(fileURLWithPath: String(cString: directory), isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }

    /// Uses `url`, just chosen in an open panel, from now on: its `hub` folder, or the folder
    /// itself when it already holds models in the hub's layout.
    static func choose(_ url: URL) throws {
        let bookmark = try url.bookmarkData(options: .withSecurityScope)
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []
        let isHub = !entries.contains("hub") && entries.contains { $0.hasPrefix("models--") }
        stopAccessing()
        _ = url.startAccessingSecurityScopedResource()
        chosen.withLock { $0 = Chosen(url: url, isHub: isHub) }
        UserDefaults.standard.set(bookmark, forKey: bookmarkKey)
        UserDefaults.standard.set(isHub, forKey: isHubKey)
        UserDefaults.standard.set(url.path(percentEncoded: false), forKey: pathKey)
    }

    /// Back to the default folder.
    static func useDefault() {
        stopAccessing()
        chosen.withLock { $0 = nil }
        for key in [bookmarkKey, isHubKey, pathKey] { UserDefaults.standard.removeObject(forKey: key) }
        prepareDefault()
    }

    /// Opens the chosen folder for the life of the app, renewing a stale bookmark; otherwise the
    /// default one is used.
    static func restore() {
        if let bookmark = UserDefaults.standard.data(forKey: bookmarkKey) {
            var isStale = false
            if let url = try? URL(resolvingBookmarkData: bookmark, options: .withSecurityScope, bookmarkDataIsStale: &isStale),
               url.startAccessingSecurityScopedResource() {
                if isStale, let renewed = try? url.bookmarkData(options: .withSecurityScope) {
                    UserDefaults.standard.set(renewed, forKey: bookmarkKey)
                }
                chosen.withLock { $0 = Chosen(url: url, isHub: UserDefaults.standard.bool(forKey: isHubKey)) }
                return
            }
        }
        prepareDefault()
    }

    /// The App Store build's folder in the container, made when missing and kept out of Time
    /// Machine backups: its models are downloads.
    private static func prepareDefault() {
        #if APPSTORE
        var home = defaultHome
        guard !FileManager.default.fileExists(atPath: home.path) else { return }
        try? FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? home.setResourceValues(values)
        #endif
    }

    private static func stopAccessing() {
        chosen.withLock { $0?.url.stopAccessingSecurityScopedResource() }
    }
}
