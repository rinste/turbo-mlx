import Foundation
import Observation

/// Owns the Python side: the virtual environment with mflux, and the long-lived worker process
/// that keeps a model loaded between generations.
@Observable
final class BackendController {
    enum Status: Equatable {
        case checking
        case notInstalled
        case installing
        case starting
        case ready
        case stopped
        case failed(String)
    }

    enum BackendError: LocalizedError {
        case notRunning

        var errorDescription: String? { "The image engine is not running." }
    }

    /// The mflux revision this app was built and tested against. Ming-Image support was merged
    /// after mflux 0.20.0, the latest release on PyPI. It is installed from GitHub's source
    /// archive rather than git+https, so the user's Mac does not need git.
    static let mfluxCommit = "5e607e29559fb6354268cf6ed41cb6891ca9afcb"
    static let mfluxRequirement = "mflux @ https://github.com/mflux-community/mflux/archive/\(mfluxCommit).zip"

    /// App data (Python environment, history). TURBO_MLX_HOME points it elsewhere, e.g. to try the
    /// first-run experience without touching the real installation.
    static let supportDirectory: URL = {
        if let custom = ProcessInfo.processInfo.environment["TURBO_MLX_HOME"], !custom.isEmpty {
            return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath, isDirectory: true)
        }
        return URL.applicationSupportDirectory.appending(path: "TurboMLX", directoryHint: .isDirectory)
    }()
    static let venvDirectory = supportDirectory.appending(path: "venv", directoryHint: .isDirectory)
    static let python = venvDirectory.appending(path: "bin/python")
    /// The standalone Python that uv downloads lives with the app's data, not in the user's home.
    static let pythonInstallDirectory = supportDirectory.appending(path: "python", directoryHint: .isDirectory)
    static let uvCacheDirectory = URL.cachesDirectory.appending(path: "TurboMLX/uv", directoryHint: .isDirectory)

    private(set) var status = Status.checking
    private(set) var info: BackendInfo?
    /// Model path currently held in memory by the worker.
    private(set) var loadedModelPath: String?
    private(set) var environment = ProcessInfo.processInfo.environment
    private(set) var installOutput: [String] = []
    /// The step the installer is on, from its "==> " lines.
    private(set) var installPhase: String?
    let log = LogBuffer()

    /// Job events (phases, progress, results) are forwarded here.
    var onEvent: ((WorkerEvent) -> Void)?
    /// The worker is ready to accept jobs.
    var onReady: (() -> Void)?
    /// The worker died on its own; the message explains it.
    var onCrash: ((String) -> Void)?

    private var worker: LineProcess?
    private var installer: LineProcess?

    var isInstalled: Bool { FileManager.default.isExecutableFile(atPath: Self.python.path) }
    /// True while the installer runs to bring an existing engine up to date.
    private(set) var isUpdating = false

    /// The mflux requirement the current environment was installed with.
    private var installedRequirement: String? {
        let marker = Self.venvDirectory.appending(path: ".turbo-mflux-requirement")
        return (try? String(contentsOf: marker, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    var isRunning: Bool { worker != nil }

    private var workerScript: URL? { Bundle.main.url(forResource: "turbo_worker", withExtension: "py") }
    private var setupScript: URL? { Bundle.main.url(forResource: "setup_backend", withExtension: "sh") }
    /// uv ships in Contents/MacOS; it also downloads Python, so the Mac needs no developer tools.
    private var bundledUV: URL? { Bundle.main.url(forAuxiliaryExecutable: "uv") }

    func prepare() async {
        environment = await ShellEnvironment.resolve()
        if !isInstalled {
            status = .notInstalled
        } else if installedRequirement != Self.mfluxRequirement {
            // A new version of the app expects a different mflux: update in place (the other
            // packages are already there, so this is quick).
            isUpdating = true
            install()
        } else {
            startWorker()
        }
    }

    // MARK: Installation

    /// Creates the Python environment with mflux, then starts the worker. `clean` rebuilds it
    /// from scratch (packages come from uv's cache, so it is quick).
    func install(clean: Bool = false) {
        guard installer == nil, let setupScript else { return }
        stopWorker()
        status = .installing
        installOutput = []
        installPhase = nil
        var environment = environment
        environment["TURBO_UV"] = bundledUV?.path
        environment["UV_PYTHON_INSTALL_DIR"] = Self.pythonInstallDirectory.path
        environment["UV_CACHE_DIR"] = Self.uvCacheDirectory.path
        environment["TURBO_REINSTALL"] = clean ? "1" : "0"
        let process = LineProcess(
            executable: URL(fileURLWithPath: "/bin/zsh"),
            arguments: [setupScript.path, Self.venvDirectory.path, Self.mfluxRequirement],
            environment: environment
        )
        do {
            try process.start(
                onStdout: { [weak self] in self?.appendInstallOutput($0) },
                onStderr: { [weak self] in self?.appendInstallOutput($0) },
                onExit: { [weak self] in self?.installFinished(status: $0) }
            )
            installer = process
        } catch {
            status = .failed("Couldn’t start the installation: \(error.localizedDescription)")
        }
    }

    private func appendInstallOutput(_ line: String) {
        if line.hasPrefix("==> ") { installPhase = String(line.dropFirst(4)) }
        installOutput.append(line)
        log.append(line)
    }

    private func installFinished(status code: Int32) {
        installer = nil
        let wasUpdate = isUpdating
        isUpdating = false
        if code == 0, isInstalled {
            startWorker()
        } else if wasUpdate, isInstalled {
            // Offline, say: keep using the engine that is there; the update is retried next launch.
            log.append("[turbo] engine update failed (exit code \(code)); keeping the installed version")
            installOutput = []
            startWorker()
        } else {
            status = .failed("The installation failed (exit code \(code)). The log has the details.")
        }
    }

    // MARK: Worker

    func startWorker() {
        guard worker == nil, installer == nil else { return }
        guard isInstalled, let workerScript else {
            status = .notInstalled
            return
        }
        status = .starting
        info = nil
        loadedModelPath = nil
        let process = LineProcess(
            executable: Self.python,
            arguments: ["-u", workerScript.path, "serve"],
            environment: environment
        )
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
                mflux: event.mflux ?? "?",
                mlx: event.mlx ?? "?",
                python: event.python ?? "?",
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
