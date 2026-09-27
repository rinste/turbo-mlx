import Foundation
import Observation

/// Which process generates the images: `turbo_worker.py` on mflux, or the native `turbo-engine`
/// (MLX Swift). Both speak the same JSON lines; one runs at a time.
nonisolated enum EngineKind: String, Sendable {
    case python
    case native

    var displayName: String {
        switch self {
        case .python: "Python engine (mflux)"
        case .native: "Native engine (MLX Swift)"
        }
    }
}

/// Owns the engines: the native `turbo-engine` the app ships with, the Python virtual environment
/// with mflux and its worker as the fallback of a build without it, and whichever of the two is
/// running now, keeping a model loaded between generations. With the native engine present,
/// Python is neither installed, updated nor started.
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

    /// Families the native engine implements: every built-in one. It is preferred for them
    /// whenever the executable is there; the Python engine only serves what it does not run.
    static let nativeFamilies: Set<ModelFamily> = [.flux2Klein, .zImageTurbo, .qwenImage, .ming]

    /// The native engine: `TURBO_ENGINE` (a development build), the one in the app bundle (put
    /// there by the "Embed turbo-engine" build phase, `scripts/embed-engine.sh`), or one dropped
    /// into the app's data folder by `scripts/build-engine.sh`.
    static var nativeEngineURL: URL? {
        var candidates: [URL] = []
        if let custom = ProcessInfo.processInfo.environment["TURBO_ENGINE"], !custom.isEmpty {
            candidates.append(URL(fileURLWithPath: (custom as NSString).expandingTildeInPath))
        }
        if let bundled = Bundle.main.url(forAuxiliaryExecutable: "turbo-engine") { candidates.append(bundled) }
        candidates.append(supportDirectory.appending(path: "bin/turbo-engine"))
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    private(set) var status = Status.checking
    /// The engine the running (or starting) worker is, nil when none runs.
    private(set) var activeKind: EngineKind?
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
    /// The Python environment exists but was installed for another mflux than this app expects.
    private var pythonNeedsUpdate: Bool { isInstalled && installedRequirement != Self.mfluxRequirement }
    var isRunning: Bool { worker != nil }

    private var workerScript: URL? { Bundle.main.url(forResource: "turbo_worker", withExtension: "py") }
    private var setupScript: URL? { Bundle.main.url(forResource: "setup_backend", withExtension: "sh") }
    /// uv ships in Contents/MacOS; it also downloads Python, so the Mac needs no developer tools.
    private var bundledUV: URL? { Bundle.main.url(forAuxiliaryExecutable: "uv") }

    var hasNativeEngine: Bool { Self.nativeEngineURL != nil }

    /// The engine a model runs on in this build: the native one for every family it implements
    /// (classifier-free guidance included: Qwen-Image, Ming-Image and Klein's base checkpoints),
    /// mflux for the rest.
    func engineKind(for model: ModelDescriptor) -> EngineKind {
        hasNativeEngine && Self.nativeFamilies.contains(model.family) ? .native : .python
    }

    /// Resolves the login shell's environment (`HF_HOME`, `HF_TOKEN`, PATH) and reports whether an
    /// engine is there. Nothing is installed or started here: that is left to `ensureWorker(for:)`,
    /// which knows which engine the selected model needs, so a build with the native engine never
    /// touches Python.
    func prepare() async {
        environment = await ShellEnvironment.resolve()
        status = hasNativeEngine || isInstalled ? .stopped : .notInstalled
    }

    /// Makes `kind` the running engine: starts it, or replaces the other one (the loaded model goes
    /// with it). The Python engine is first brought up to date when a new version of the app
    /// expects a different mflux (in place: the other packages are already there, so it is
    /// quick); when it is not installed, reports that instead.
    func ensureWorker(for kind: EngineKind) {
        if worker != nil, activeKind == kind { return }
        guard installer == nil else { return }
        if kind == .python, pythonNeedsUpdate {
            isUpdating = true
            install()
            return
        }
        if worker != nil { stopWorker() }
        startWorker(kind: kind)
    }

    // MARK: Installation

    /// Creates the Python environment with mflux, then starts an engine. `clean` rebuilds it from
    /// scratch (packages come from uv's cache, so it is quick).
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
        // The Python engine was just installed for a build without the native one; after a repair
        // in a build that has it, the native engine comes back.
        let kind: EngineKind = hasNativeEngine ? .native : .python
        if code == 0, isInstalled {
            startWorker(kind: kind)
        } else if wasUpdate, isInstalled {
            // Offline, say: keep using the engine that is there; the update is retried next time.
            log.append("[turbo] engine update failed (exit code \(code)); keeping the installed version")
            installOutput = []
            startWorker(kind: kind)
        } else {
            status = .failed("The installation failed (exit code \(code)). The log has the details.")
        }
    }

    // MARK: Worker

    func startWorker(kind: EngineKind = .python) {
        guard worker == nil, installer == nil else { return }
        let process: LineProcess
        switch kind {
        case .python:
            guard isInstalled, let workerScript else {
                status = .notInstalled
                return
            }
            process = LineProcess(
                executable: Self.python,
                arguments: ["-u", workerScript.path, "serve"],
                environment: environment
            )
        case .native:
            guard let engine = Self.nativeEngineURL else {
                status = .failed("The native engine is not available in this build.")
                return
            }
            process = LineProcess(executable: engine, arguments: ["serve"], environment: environment)
        }
        activeKind = kind
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
        activeKind = nil
        loadedModelPath = nil
        worker.stop(grace: .seconds(2))
        if status == .ready || status == .starting { status = .stopped }
    }

    func restartWorker() {
        let kind = activeKind ?? (hasNativeEngine ? .native : .python)
        stopWorker()
        startWorker(kind: kind)
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
                engine: event.engine ?? event.mflux.map { "mflux \($0)" } ?? "?",
                runtime: event.python.map { "Python \($0)" } ?? "Swift",
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
        activeKind = nil
        loadedModelPath = nil
        let lastLines = log.tail(3)
        let message = "The image engine quit unexpectedly (exit code \(code))."
            + (lastLines.isEmpty ? "" : "\n\n\(lastLines)")
        status = .failed(message)
        onCrash?(message)
    }
}
