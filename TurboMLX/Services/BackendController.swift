import Foundation
import Observation

/// Owns the image engine: `turbo-engine`, the MLX Swift executable the app ships with, run as a
/// child process that speaks JSON lines and keeps a model loaded between generations.
@Observable
final class BackendController {
    enum Status: Equatable {
        case checking
        case starting
        case ready
        case stopped
        case failed(String)
    }

    enum BackendError: LocalizedError {
        case notRunning

        var errorDescription: String? { "The image engine is not running." }
    }

    /// App data (history). TURBO_MLX_HOME points it elsewhere, e.g. to try the first-run experience
    /// without touching the real installation.
    static let supportDirectory: URL = {
        if let custom = ProcessInfo.processInfo.environment["TURBO_MLX_HOME"], !custom.isEmpty {
            return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath, isDirectory: true)
        }
        return URL.applicationSupportDirectory.appending(path: "TurboMLX", directoryHint: .isDirectory)
    }()

    /// The engine: `TURBO_ENGINE` (a development build) or the one in the app bundle, put there by
    /// the "Embed turbo-engine" build phase (`scripts/embed-engine.sh`).
    static var engineURL: URL? {
        var candidates: [URL] = []
        if let custom = ProcessInfo.processInfo.environment["TURBO_ENGINE"], !custom.isEmpty {
            candidates.append(URL(fileURLWithPath: (custom as NSString).expandingTildeInPath))
        }
        if let bundled = Bundle.main.url(forAuxiliaryExecutable: "turbo-engine") { candidates.append(bundled) }
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private(set) var status = Status.checking
    private(set) var info: BackendInfo?
    /// Model path currently held in memory by the engine.
    private(set) var loadedModelPath: String?
    private(set) var environment = ProcessInfo.processInfo.environment
    let log = LogBuffer()

    /// Job events (phases, progress, results) are forwarded here.
    var onEvent: ((WorkerEvent) -> Void)?
    /// The engine is ready to accept jobs.
    var onReady: (() -> Void)?
    /// The engine died on its own; the message explains it.
    var onCrash: ((String) -> Void)?

    private var worker: LineProcess?

    var isRunning: Bool { worker != nil }

    /// Resolves the login shell's environment (`HF_HOME`, `HF_TOKEN`), which the model cache and
    /// the downloads follow. The engine is started by `ensureWorker()`.
    func prepare() async {
        environment = await ShellEnvironment.resolve()
        status = .stopped
    }

    /// Starts the engine unless it is running.
    func ensureWorker() {
        if worker == nil { startWorker() }
    }

    // MARK: Worker

    func startWorker() {
        guard worker == nil else { return }
        guard let engine = Self.engineURL else {
            status = .failed("The image engine is missing from this copy of Turbo MLX. Install the app again.")
            return
        }
        let process = LineProcess(executable: engine, arguments: ["serve"], environment: environment)
        status = .starting
        info = nil
        loadedModelPath = nil
        do {
            try process.start(
                onStdout: { [weak self, weak process] line in
                    // A worker that was replaced (restart, force stop) no longer speaks for the app.
                    guard let self, let process, worker === process else { return }
                    handle(line: line)
                },
                onStderr: { [weak self] in self?.log.append($0) },
                onExit: { [weak self] in self?.workerExited(process, status: $0) }
            )
            worker = process
        } catch {
            status = .failed("Couldn’t start the image engine: \(error.localizedDescription)")
        }
    }

    /// Stops the worker right away, even in the middle of a generation.
    func stopWorker() {
        guard let worker else { return }
        self.worker = nil
        loadedModelPath = nil
        worker.stop(grace: .seconds(2))
        if status == .ready || status == .starting { status = .stopped }
    }

    func restartWorker() {
        stopWorker()
        startWorker()
    }

    func send(_ command: WorkerCommand) throws {
        guard let worker else { throw BackendError.notRunning }
        try worker.send(line: command.jsonLine)
    }

    func unloadModel() {
        try? send(.unload)
    }

    private func handle(line: String) {
        guard let event = WorkerEvent.parse(line) else {
            log.append(line)
            return
        }
        switch event.event {
        case "ready":
            info = BackendInfo(
                engine: event.engine ?? "turbo-engine",
                mlx: event.mlx ?? "?",
                device: event.device ?? "Apple Silicon",
                memory: event.memory ?? 0
            )
            status = .ready
            onReady?()
        case "model_loaded":
            loadedModelPath = event.path
            onEvent?(event)
        case "unloaded":
            loadedModelPath = nil
        case "fatal":
            log.append(event.message ?? "Fatal engine error")
        default:
            onEvent?(event)
        }
    }

    private func workerExited(_ process: LineProcess, status code: Int32) {
        // Exits of workers we stopped on purpose are not news.
        guard worker === process else { return }
        worker = nil
        loadedModelPath = nil
        let lastLines = log.tail(3)
        let message = "The image engine quit unexpectedly (exit code \(code))."
            + (lastLines.isEmpty ? "" : "\n\n\(lastLines)")
        status = .failed(message)
        onCrash?(message)
    }
}
