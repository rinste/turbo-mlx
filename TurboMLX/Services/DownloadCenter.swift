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

    private var tasks: [String: Task<Void, Never>] = [:]
    private var samples: [String: (date: Date, bytes: Int64)] = [:]
    /// Companions being fetched. The LTX-2 packs share one: two downloads of a file would write
    /// into the same partial file, so the second waits and then finds it cached.
    private var companionsInFlight: Set<String> = []

    private struct Part {
        let downloader: HubDownloader
        let patterns: [String]
        /// The companion's repository, for a part other models share.
        let companion: String?
    }

    func isDownloading(_ model: ModelDescriptor) -> Bool { active[model.id] != nil }

    /// Downloads the model's repository and, for a family that needs one, its companion (LTX-2's
    /// text encoder), as one download: the files already in the cache count as done.
    func start(_ model: ModelDescriptor, hubCache: URL) {
        guard let repo = model.repo, tasks[model.id] == nil else { return }
        failures[model.id] = nil
        active[model.id] = Progress(total: model.sizeBytes ?? 0)
        var parts = [Part(downloader: HubDownloader(repo: repo, hubCache: hubCache), patterns: model.family.downloadPatterns,
                          companion: nil)]
        if let companion = model.family.companion {
            parts.append(Part(downloader: HubDownloader(repo: companion.repo, hubCache: hubCache), patterns: companion.patterns,
                              companion: companion.repo))
        }
        tasks[model.id] = Task { [weak self] in
            do {
                var listings: [(part: Part, listing: HubDownloader.Listing)] = []
                for part in parts {
                    listings.append((part, try await part.downloader.list(matching: part.patterns)))
                }
                self?.update(model, total: listings.reduce(0) { $0 + $1.listing.totalBytes })
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
            } catch {
                if !Task.isCancelled { self?.failures[model.id] = error.localizedDescription }
            }
            self?.finished(model)
        }
    }

    func cancel(_ model: ModelDescriptor) {
        guard let task = tasks.removeValue(forKey: model.id) else { return }
        task.cancel()
        active[model.id] = nil
        samples[model.id] = nil
    }

    func cancelAll() {
        for task in tasks.values { task.cancel() }
        tasks.removeAll()
        active.removeAll()
        samples.removeAll()
    }

    private func update(_ model: ModelDescriptor, total: Int64) {
        guard var progress = active[model.id] else { return }
        progress.total = total
        active[model.id] = progress
    }

    private func update(_ model: ModelDescriptor, bytes: Int64) {
        guard var progress = active[model.id] else { return }
        let now = Date()
        if let previous = samples[model.id] {
            let elapsed = now.timeIntervalSince(previous.date)
            if elapsed >= 1 {
                let instant = Double(bytes - previous.bytes) / elapsed
                // Smooth the rate so the remaining time does not jump around.
                progress.bytesPerSecond = progress.bytesPerSecond == 0 ? instant : progress.bytesPerSecond * 0.8 + instant * 0.2
                samples[model.id] = (now, bytes)
            }
        } else {
            samples[model.id] = (now, bytes)
        }
        progress.bytes = bytes
        active[model.id] = progress
    }

    private func finished(_ model: ModelDescriptor) {
        // A cancelled download was already removed.
        guard tasks[model.id] != nil else { return }
        tasks[model.id] = nil
        active[model.id] = nil
        samples[model.id] = nil
        onFinish?(model)
    }
}
