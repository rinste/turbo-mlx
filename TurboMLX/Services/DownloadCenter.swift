import Foundation
import Observation

/// Downloads Hugging Face models into the hub cache, in the app's own process, so a model can be
/// fetched before (or without) any engine.
@Observable
final class DownloadCenter {
    struct Progress: Equatable {
        var bytes: Int64 = 0
        var total: Int64 = 0
        var bytesPerSecond: Double = 0
        let startedAt = Date()

        var fraction: Double? { total > 0 ? min(1, Double(bytes) / Double(total)) : nil }

        var secondsRemaining: Double? {
            guard bytesPerSecond > 1, total > bytes else { return nil }
            return Double(total - bytes) / bytesPerSecond
        }
    }

    private(set) var active: [String: Progress] = [:]
    private(set) var failures: [String: String] = [:]
    /// Called on the main actor when a download ends, successfully or not (not when cancelled).
    var onFinish: ((ModelDescriptor) -> Void)?
    /// The same for a model's Quality files (`startQuality`).
    var onQualityFinish: ((ModelDescriptor) -> Void)?

    private var tasks: [String: Task<Void, Never>] = [:]
    private var samples: [String: (date: Date, bytes: Int64)] = [:]
    /// Companions being fetched. The LTX-2 packs share one: two downloads of a file would write
    /// into the same partial file, so the second waits and then finds it cached.
    private var companionsInFlight: Set<String> = []
    /// Asleep, the Mac would drop the connection of a download that takes the better part of an hour.
    @ObservationIgnored private let awake = KeepAwake(reason: "Downloading a model")

    private struct Part {
        let downloader: HubDownloader
        let patterns: [String]
        /// The companion's repository, for a part other models share.
        let companion: String?
    }

    /// Why a download stopped although Hugging Face answered.
    nonisolated enum Problem: LocalizedError {
        case notEnoughSpace(needed: Int64, free: Int64)
        case incomplete(family: String)

        var errorDescription: String? {
            switch self {
            case .notEnoughSpace(let needed, let free):
                "It needs \(Self.bytes(needed)) more, but only \(Self.bytes(free)) are free. Free up some space and try again: what was already downloaded is kept."
            case .incomplete(let family):
                "The files arrived, but they don’t make a complete \(family) model: the repository may not be in mflux format."
            }
        }

        private static func bytes(_ value: Int64) -> String {
            ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
        }
    }

    func isDownloading(_ model: ModelDescriptor) -> Bool { active[model.id] != nil }

    /// Forgets why the model's last download failed (its files went to the Trash since).
    func clearFailure(_ model: ModelDescriptor) {
        failures[model.id] = nil
    }

    /// Downloads the model's repository and, for a family that needs one, its companion (LTX-2's
    /// text encoder), as one download: the files already in the cache count as done. A catalog
    /// model comes at the commit it was checked with. Stops before starting when the disk cannot
    /// take what is missing, and fails when what arrived is not a model `locator` recognizes.
    func start(_ model: ModelDescriptor, locator: ModelLocator) {
        guard let repo = model.repo, tasks[model.id] == nil else { return }
        failures[model.id] = nil
        active[model.id] = Progress(total: model.sizeBytes ?? 0)
        let hubCache = locator.hubCache
        var parts = [Part(downloader: HubDownloader(repo: repo, hubCache: hubCache, revision: model.revision ?? "main"),
                          patterns: model.family.downloadPatterns, companion: nil)]
        if let companion = model.family.companion {
            parts.append(Part(downloader: HubDownloader(repo: companion.repo, hubCache: hubCache, revision: companion.revision),
                              patterns: companion.patterns, companion: companion.repo))
        }
        tasks[model.id] = Task { [weak self] in
            do {
                var listings: [(part: Part, listing: HubDownloader.Listing)] = []
                for part in parts {
                    listings.append((part, try await part.downloader.list(matching: part.patterns)))
                }
                self?.update(model, total: listings.reduce(0) { $0 + $1.listing.totalBytes })
                try Self.checkSpace(for: listings.reduce(0) { $0 + $1.part.downloader.bytesMissing(from: $1.listing) },
                                    in: hubCache)
                var done: Int64 = 0
                for (part, listing) in listings {
                    let base = done
                    if let companion = part.companion {
                        while self?.companionsInFlight.contains(companion) == true { try await Task.sleep(for: .seconds(1)) }
                        self?.companionsInFlight.insert(companion)
                    }
                    defer { if let companion = part.companion { self?.companionsInFlight.remove(companion) } }
                    // Its own weak capture: the Sendable closure may not read the task's `self` var.
                    _ = try await part.downloader.download(listing) { [weak self] bytes in
                        Task { @MainActor [weak self] in self?.update(model, bytes: base + bytes) }
                    }
                    done += listing.totalBytes
                }
                // Files renamed upstream, or a repository added by hand that is not an mflux
                // checkpoint, would otherwise end in silence, with the model still not downloaded.
                if locator.installedLocation(of: model) == nil {
                    throw Problem.incomplete(family: model.family.displayName)
                }
            } catch {
                if !Task.isCancelled { self?.failures[model.id] = error.localizedDescription }
            }
            self?.finished(model)
        }
        awake.isOn = true
    }

    func cancel(_ model: ModelDescriptor) {
        guard let task = tasks.removeValue(forKey: model.id) else { return }
        task.cancel()
        active[model.id] = nil
        samples[model.id] = nil
        awake.isOn = !tasks.isEmpty
    }

    func cancelAll() {
        for task in tasks.values { task.cancel() }
        tasks.removeAll()
        active.removeAll()
        samples.removeAll()
        awake.isOn = false
    }

    /// Stops before a download that would fill the disk, leaving 2 GB to the system.
    private static func checkSpace(for needed: Int64, in folder: URL) throws {
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        guard needed > 0,
              let free = (try? folder.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
                  .volumeAvailableCapacityForImportantUsage,
              free < needed + (2 << 30)
        else { return }
        throw Problem.notEnoughSpace(needed: needed, free: free)
    }

    private func update(_ model: ModelDescriptor, total: Int64) { update(key: model.id, total: total) }
    private func update(_ model: ModelDescriptor, bytes: Int64) { update(key: model.id, bytes: bytes) }
    private func finished(_ model: ModelDescriptor) { finished(model, key: model.id) }

    private func update(key: String, total: Int64) {
        guard var progress = active[key] else { return }
        progress.total = total
        active[key] = progress
    }

    private func update(key: String, bytes: Int64) {
        guard var progress = active[key] else { return }
        let now = Date()
        if let previous = samples[key] {
            let elapsed = now.timeIntervalSince(previous.date)
            if elapsed >= 1 {
                let instant = Double(bytes - previous.bytes) / elapsed
                // Smooth the rate so the remaining time does not jump around.
                progress.bytesPerSecond = progress.bytesPerSecond == 0 ? instant : progress.bytesPerSecond * 0.8 + instant * 0.2
                samples[key] = (now, bytes)
            }
        } else {
            samples[key] = (now, bytes)
        }
        progress.bytes = bytes
        active[key] = progress
    }

    private func finished(_ model: ModelDescriptor, key: String) {
        // A cancelled download was already removed.
        guard tasks[key] != nil else { return }
        tasks[key] = nil
        active[key] = nil
        samples[key] = nil
        awake.isOn = !tasks.isEmpty
        if key == model.id { onFinish?(model) } else { onQualityFinish?(model) }
    }

    // MARK: Quality files

    /// The key the Quality mode's files of a model download under, apart from the model's own.
    static func qualityKey(_ model: ModelDescriptor) -> String { model.id + "#quality" }

    func qualityProgress(_ model: ModelDescriptor) -> Progress? { active[Self.qualityKey(model)] }
    func qualityFailure(_ model: ModelDescriptor) -> String? { failures[Self.qualityKey(model)] }

    /// Downloads the files only the Quality mode needs (`ModelFamily.qualityFiles`) into the
    /// installed model's snapshot, at the model's commit.
    func startQuality(_ model: ModelDescriptor, locator: ModelLocator) {
        let key = Self.qualityKey(model)
        let files = model.family.qualityFiles
        guard let repo = model.repo, tasks[key] == nil, !files.isEmpty else { return }
        failures[key] = nil
        active[key] = Progress(total: model.qualitySizeBytes ?? 0)
        let hubCache = locator.hubCache
        let downloader = HubDownloader(repo: repo, hubCache: hubCache, revision: model.revision ?? "main")
        tasks[key] = Task { [weak self] in
            do {
                let listing = try await downloader.list(matching: files)
                self?.update(key: key, total: listing.totalBytes)
                try Self.checkSpace(for: downloader.bytesMissing(from: listing), in: hubCache)
                _ = try await downloader.download(listing) { [weak self] bytes in
                    Task { @MainActor [weak self] in self?.update(key: key, bytes: bytes) }
                }
            } catch {
                if !Task.isCancelled { self?.failures[key] = error.localizedDescription }
            }
            self?.finished(model, key: key)
        }
        awake.isOn = true
    }

    func cancelQuality(_ model: ModelDescriptor) {
        let key = Self.qualityKey(model)
        guard let task = tasks.removeValue(forKey: key) else { return }
        task.cancel()
        active[key] = nil
        samples[key] = nil
        awake.isOn = !tasks.isEmpty
    }
}
