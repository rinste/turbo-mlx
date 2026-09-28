import Foundation
import Observation

/// Asks GitHub for the latest release of Turbo MLX (drafts and pre-releases excluded): at launch at
/// most once a day while automatic checks are on, and from Turbo MLX → Check for Updates. The
/// request carries nothing about the Mac, its prompts or its images.
@Observable
final class UpdateChecker {
    nonisolated struct Release: Equatable, Sendable {
        let version: String
        let page: URL
    }

    nonisolated enum CheckError: LocalizedError {
        case status(Int)
        case unreadable

        var errorDescription: String? {
            switch self {
            case .status(404): "GitHub has no published release of Turbo MLX."
            case .status(let status): "GitHub answered \(status)."
            case .unreadable: "GitHub’s answer could not be read."
            }
        }
    }

    nonisolated static let latestReleaseURL = URL(string: "https://api.github.com/repos/rinste/turbo-mlx/releases/latest")!

    /// A release newer than this copy, found by the last check.
    private(set) var newer: Release?

    var checksAutomatically: Bool {
        didSet { UserDefaults.standard.set(checksAutomatically, forKey: Keys.automatic) }
    }

    private enum Keys {
        static let automatic = "checksForUpdates"
        static let lastCheck = "lastUpdateCheck"
        static let announced = "announcedUpdateVersion"
    }

    init() {
        checksAutomatically = UserDefaults.standard.object(forKey: Keys.automatic) as? Bool ?? true
    }

    static var currentVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    /// Automatic checks are on and the last one is more than a day old.
    var isDue: Bool {
        guard checksAutomatically else { return false }
        let last = UserDefaults.standard.object(forKey: Keys.lastCheck) as? Date ?? .distantPast
        return Date().timeIntervalSince(last) > 24 * 60 * 60
    }

    /// The latest release when it is newer than this copy, nil when this copy is the latest.
    func check() async throws -> Release? {
        var request = URLRequest(url: Self.latestReleaseURL, timeoutInterval: 30)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("turbo-mlx", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { throw CheckError.status(status) }
        guard let release = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tag = release["tag_name"] as? String,
              let page = (release["html_url"] as? String).flatMap(URL.init(string:))
        else { throw CheckError.unreadable }
        UserDefaults.standard.set(Date(), forKey: Keys.lastCheck)
        let version = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
        newer = Self.isNewer(version, than: Self.currentVersion) ? Release(version: version, page: page) : nil
        return newer
    }

    /// True the first time a release is announced: a check at launch mentions each version once.
    func announcesOnce(_ release: Release) -> Bool {
        guard UserDefaults.standard.string(forKey: Keys.announced) != release.version else { return false }
        UserDefaults.standard.set(release.version, forKey: Keys.announced)
        return true
    }

    /// Compares dotted versions number by number: "1.10" is newer than "1.9", "1.0.1" than "1.0".
    nonisolated static func isNewer(_ candidate: String, than current: String) -> Bool {
        let lhs = candidate.split(separator: ".").map { Int($0) ?? 0 }
        let rhs = current.split(separator: ".").map { Int($0) ?? 0 }
        for index in 0..<max(lhs.count, rhs.count) {
            let left = index < lhs.count ? lhs[index] : 0
            let right = index < rhs.count ? rhs[index] : 0
            if left != right { return left > right }
        }
        return false
    }
}
