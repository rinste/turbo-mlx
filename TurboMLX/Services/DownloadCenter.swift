import Foundation
import Observation

/// Downloads Hugging Face models into the hub cache through `turbo_worker.py download`.
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
    /// Called on the main actor when a download ends, successfully or not.
    var onFinish: ((ModelDescriptor) -> Void)?

    private var processes: [String: LineProcess] = [:]
    private var samples: [String: (date: Date, bytes: Int64)] = [:]

    func isDownloading(_ model: ModelDescriptor) -> Bool { active[model.id] != nil }

    func start(_ model: ModelDescriptor, backend: BackendController) {
        guard let repo = model.repo, processes[model.id] == nil,
              let script = Bundle.main.url(forResource: "turbo_worker", withExtension: "py")
        else { return }
        failures[model.id] = nil
        active[model.id] = Progress(total: model.sizeBytes ?? 0)
        let process = LineProcess(
            executable: BackendController.python,
            arguments: ["-u", script.path, "download", "--repo", repo, "--include"] + model.family.downloadPatterns,
            environment: backend.environment
        )
        do {
            try process.start(
                onStdout: { [weak self] in self?.handle(line: $0, for: model) },
                onStderr: { [weak backend] in backend?.log.append($0) },
                onExit: { [weak self] in self?.finished(model, process: process, status: $0) }
            )
            processes[model.id] = process
        } catch {
            active[model.id] = nil
            failures[model.id] = error.localizedDescription
        }
    }

    func cancel(_ model: ModelDescriptor) {
        guard let process = processes.removeValue(forKey: model.id) else { return }
        process.stop()
        active[model.id] = nil
        samples[model.id] = nil
    }

    func cancelAll() {
        for process in processes.values { process.stop() }
        processes.removeAll()
        active.removeAll()
    }

    private func handle(line: String, for model: ModelDescriptor) {
        guard let event = WorkerEvent.parse(line), var progress = active[model.id] else { return }
        switch event.event {
        case "download_started":
            progress.total = event.total ?? progress.total
            progress.bytes = event.cached ?? 0
        case "download_progress":
            let bytes = event.bytes ?? progress.bytes
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
            progress.total = event.total ?? progress.total
        case "failed":
            failures[model.id] = event.message
        default:
            break
        }
        active[model.id] = progress
    }

    private func finished(_ model: ModelDescriptor, process: LineProcess, status: Int32) {
        // A cancelled download was already removed.
        guard processes[model.id] === process else { return }
        processes[model.id] = nil
        active[model.id] = nil
        samples[model.id] = nil
        if status != 0, failures[model.id] == nil {
            failures[model.id] = "The download stopped (exit code \(status))."
        }
        onFinish?(model)
    }
}
