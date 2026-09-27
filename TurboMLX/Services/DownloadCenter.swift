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

    func isDownloading(_ model: ModelDescriptor) -> Bool { active[model.id] != nil }

    func start(_ model: ModelDescriptor, hubCache: URL, environment: [String: String]) {
        guard let repo = model.repo, tasks[model.id] == nil else { return }
        failures[model.id] = nil
        active[model.id] = Progress(total: model.sizeBytes ?? 0)
        let downloader = HubDownloader(repo: repo, hubCache: hubCache, environment: environment)
        let patterns = model.family.downloadPatterns
        tasks[model.id] = Task { [weak self] in
            do {
                let listing = try await downloader.list(matching: patterns)
                self?.update(model, total: listing.totalBytes)
                // Its own weak capture: the Sendable closure may not read the task's `self` var.
                _ = try await downloader.download(listing) { [weak self] bytes in
                    Task { @MainActor [weak self] in self?.update(model, bytes: bytes) }
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
