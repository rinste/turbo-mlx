import AppKit
import Foundation
import Observation

struct AppAlert: Identifiable {
    let id = UUID()
    var title: String
    var message: String
    var offersLog = false
}

/// App-wide state: model catalog, generation settings, the job queue and the history.
@Observable
final class AppModel {
    /// What the image viewer shows: the job in progress / latest result, or a picked history item.
    enum ViewerSelection: Hashable {
        case live
        case item(HistoryItem.ID)
    }

    /// Why "Generate" is not available right now.
    enum Blocker: Equatable {
        case noModel
        case modelNotDownloaded
        case modelDownloading
        case backendNotInstalled
        case backendInstalling
        case emptyPrompt

        var hint: String {
            switch self {
            case .noModel: "Choose a model."
            case .modelNotDownloaded: "Download the model to get started."
            case .modelDownloading: "Waiting for the download to finish."
            case .backendNotInstalled: "Install the image engine to get started."
            case .backendInstalling: "Installing the image engine…"
            case .emptyPrompt: "Write a prompt."
            }
        }
    }

    let backend = BackendController()
    let downloads = DownloadCenter()
    let history = HistoryStore()

    private(set) var models: [ModelDescriptor]
    private(set) var installed: [String: URL] = [:]
    private(set) var locator = ModelLocator(environment: ProcessInfo.processInfo.environment)

    var selectedModelID: String {
        didSet {
            UserDefaults.standard.set(selectedModelID, forKey: Keys.selectedModel)
            if selectedModelID != oldValue {
                preloadDeclined = false
                preloadIfUseful()
            }
            guard selectedModelID != oldValue, let model = selectedModel,
                  let previous = models.first(where: { $0.id == oldValue }),
                  previous.family != model.family || previous.defaultSteps != model.defaultSteps
            else { return }
            // Steps and guidance mean different things across families: start from the new defaults.
            settings.steps = model.defaultSteps
            settings.guidance = model.defaultGuidance
        }
    }

    var settings: GenerationSettings {
        didSet {
            guard settings != oldValue else { return }
            if let data = try? JSONEncoder().encode(settings) {
                UserDefaults.standard.set(data, forKey: Keys.settings)
            }
            preloadIfUseful()
        }
    }

    /// Model path the worker was asked to load ahead of time, so the request is not repeated.
    @ObservationIgnored private var preloadRequested: String?
    /// Set when the user frees the memory: no preloading until the model changes or an image runs.
    @ObservationIgnored private var preloadDeclined = false

    private(set) var queue: [GenerationJob] = []
    private(set) var activeJob: GenerationJob?
    var viewer = ViewerSelection.live
    var alert: AppAlert?
    /// Model to generate with as soon as its download completes ("Download and Generate").
    private(set) var generateAfterDownload: String?
    @ObservationIgnored private var started = false

    private enum Keys {
        static let settings = "generationSettings"
        static let selectedModel = "selectedModelID"
        static let customModels = "customModels"
    }

    init() {
        let defaults = UserDefaults.standard
        let custom = defaults.data(forKey: Keys.customModels)
            .flatMap { try? JSONDecoder().decode([ModelDescriptor].self, from: $0) } ?? []
        let catalog = ModelCatalog.builtIn + custom
        let selectedID = defaults.string(forKey: Keys.selectedModel) ?? ModelCatalog.defaultModelID
        models = catalog
        selectedModelID = selectedID

        if let data = defaults.data(forKey: Keys.settings),
           let saved = try? JSONDecoder().decode(GenerationSettings.self, from: data) {
            settings = saved
        } else {
            var initial = GenerationSettings()
            // At 1024 px with everything resident Ming-Image peaks at ~35 GB and Qwen-Image at
            // ~43 GB (~15 and ~21 GB with Save memory): below 64 GB the full-speed mode would swap.
            initial.lowMemory = ProcessInfo.processInfo.physicalMemory < 64 << 30
            if let model = catalog.first(where: { $0.id == selectedID }) {
                initial.steps = model.defaultSteps
                initial.guidance = model.defaultGuidance
            }
            settings = initial
        }

        backend.onReady = { [weak self] in
            self?.preloadRequested = nil
            self?.pump()
            self?.preloadIfUseful()
        }
        backend.onEvent = { [weak self] in self?.handle($0) }
        backend.onCrash = { [weak self] in self?.backendCrashed($0) }
        downloads.onFinish = { [weak self] in self?.downloadFinished($0) }
        refreshInstalled()
    }

    func start() async {
        guard !started else { return }
        started = true
        history.load()
        await backend.prepare()
        locator = ModelLocator(environment: backend.environment)
        refreshInstalled()
        startEngineForSelection()
    }

    /// The engine the selected model runs on.
    var selectedEngine: EngineKind {
        selectedModel.map { backend.engineKind(for: $0) } ?? .python
    }

    /// Brings up the engine of the selected model, unless an installation is running.
    private func startEngineForSelection() {
        guard backend.status != .installing else { return }
        backend.ensureWorker(for: selectedEngine)
    }

    func shutdown() {
        downloads.cancelAll()
        backend.stopWorker()
    }

    // MARK: Models

    var selectedModel: ModelDescriptor? {
        models.first { $0.id == selectedModelID } ?? models.first
    }

    func isInstalled(_ model: ModelDescriptor) -> Bool { installed[model.id] != nil }

    func isLoaded(_ model: ModelDescriptor) -> Bool {
        guard let path = backend.loadedModelPath else { return false }
        return installed[model.id]?.path == path
    }

    func refreshInstalled() {
        var found: [String: URL] = [:]
        for model in models {
            if let url = locator.installedLocation(of: model) { found[model.id] = url }
        }
        installed = found
    }

    func download(_ model: ModelDescriptor) {
        if let size = model.sizeBytes, let free = freeDiskSpace(), free < size + (2 << 30) {
            alert = AppAlert(
                title: "Not enough disk space",
                message: "\(model.name) needs \(Format.bytes(size)), but only \(Format.bytes(free)) are free. Free up some space and try again."
            )
            return
        }
        downloads.start(model, hubCache: locator.hubCache, environment: backend.environment)
    }

    /// Downloads the selected model, then generates with the current prompt.
    func downloadAndGenerate() {
        guard let model = selectedModel else { return }
        generateAfterDownload = settings.trimmedPrompt.isEmpty ? nil : model.id
        download(model)
        if !downloads.isDownloading(model) { generateAfterDownload = nil }
    }

    func cancelDownload(_ model: ModelDescriptor) {
        if generateAfterDownload == model.id { generateAfterDownload = nil }
        downloads.cancel(model)
    }

    private func downloadFinished(_ model: ModelDescriptor) {
        refreshInstalled()
        let generateNow = generateAfterDownload == model.id
        if generateNow { generateAfterDownload = nil }
        if let failure = downloads.failures[model.id] {
            alert = AppAlert(title: "Couldn’t download \(model.name)", message: failure, offersLog: true)
        } else if generateNow, selectedModelID == model.id {
            generate()
        } else {
            preloadIfUseful()
        }
        if !NSApp.isActive { NSApp.requestUserAttention(.informationalRequest) }
    }

    /// Frees the loaded model and stops loading it ahead of time until the model changes or an
    /// image is generated.
    func freeMemory() {
        preloadDeclined = true
        preloadRequested = nil
        backend.unloadModel()
    }

    /// Loads the selected model as soon as a prompt is being written, so that the first image
    /// starts at the prompt instead of at a 5–40 s load. Nothing happens while the worker is
    /// busy, when the model is already in memory, or after "Free Memory". A prompt for a model
    /// of the other engine brings that engine up first.
    private func preloadIfUseful() {
        guard !preloadDeclined, activeJob == nil, queue.isEmpty, !settings.trimmedPrompt.isEmpty,
              let model = selectedModel, let location = installed[model.id]
        else { return }
        let kind = backend.engineKind(for: model)
        if backend.activeKind != kind {
            switch backend.status {
            case .ready, .stopped, .failed: backend.ensureWorker(for: kind) // onReady comes back here
            default: break
            }
            return
        }
        guard backend.status == .ready, backend.loadedModelPath != location.path, preloadRequested != location.path
        else { return }
        preloadRequested = location.path
        try? backend.send(.load(model: model, modelPath: location.path, lowMemory: settings.lowMemory))
    }

    /// Free space on the volume that holds the Hugging Face cache (which may not exist yet).
    private func freeDiskSpace() -> Int64? {
        var url = locator.hubCache
        while !FileManager.default.fileExists(atPath: url.path), url.pathComponents.count > 1 {
            url.deleteLastPathComponent()
        }
        return (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
    }

    func addCustomModel(_ model: ModelDescriptor) {
        models.removeAll { $0.id == model.id && !$0.isBuiltIn }
        guard !models.contains(where: { $0.id == model.id }) else {
            selectedModelID = model.id
            return
        }
        models.append(model)
        saveCustomModels()
        refreshInstalled()
        selectedModelID = model.id
    }

    func removeCustomModel(_ model: ModelDescriptor) {
        guard !model.isBuiltIn else { return }
        cancelDownload(model)
        models.removeAll { $0.id == model.id }
        saveCustomModels()
        if selectedModelID == model.id { selectedModelID = ModelCatalog.defaultModelID }
    }

    private func saveCustomModels() {
        let custom = models.filter { !$0.isBuiltIn }
        UserDefaults.standard.set(try? JSONEncoder().encode(custom), forKey: Keys.customModels)
    }

    // MARK: Generation

    var blocker: Blocker? {
        guard let model = selectedModel else { return .noModel }
        if backend.engineKind(for: model) == .python {
            if backend.status == .installing { return .backendInstalling }
            if !backend.isInstalled { return .backendNotInstalled }
        }
        if downloads.isDownloading(model) { return .modelDownloading }
        if !isInstalled(model) { return .modelNotDownloaded }
        if settings.trimmedPrompt.isEmpty { return .emptyPrompt }
        return nil
    }

    /// The running job first, then the queue.
    var pendingJobs: [GenerationJob] {
        (activeJob.map { [$0] } ?? []) + queue
    }

    var isBusy: Bool { activeJob != nil || !queue.isEmpty }

    func generate() {
        guard blocker == nil, let model = selectedModel else { return }
        preloadDeclined = false
        for seed in settings.nextSeeds() {
            let request = GenerationRequest(
                prompt: settings.trimmedPrompt,
                blocks: settings.blocks.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty },
                seed: seed,
                size: settings.size,
                steps: min(max(settings.steps, model.stepRange.lowerBound), model.stepRange.upperBound),
                guidance: settings.guidance,
                transparentBackground: settings.transparentBackground && model.family.producesAlpha,
                lowMemory: settings.lowMemory
            )
            queue.append(GenerationJob(model: model, request: request, outputURL: history.newImageURL(seed: seed)))
        }
        viewer = .live
        pump()
    }

    func cancel(_ job: GenerationJob) {
        if let index = queue.firstIndex(where: { $0 === job }) {
            queue.remove(at: index)
            updateDockBadge()
            return
        }
        guard job === activeJob else { return }
        if job.isCancelling {
            // Asked twice: the worker is inside a step that cannot be interrupted (loading,
            // prompt encoding, decoding). Restart it; the model will be reloaded next time.
            activeJob = nil
            backend.restartWorker()
            updateDockBadge()
        } else {
            job.isCancelling = true
            try? backend.send(.cancel(jobID: job.id))
        }
    }

    func cancelAll() {
        queue.removeAll()
        if let activeJob { cancel(activeJob) }
        updateDockBadge()
    }

    private func pump() {
        defer { updateDockBadge() }
        guard activeJob == nil, let job = queue.first else { return }
        let kind = backend.engineKind(for: job.model)
        switch backend.status {
        case .ready where backend.activeKind == kind:
            break
        case .starting, .installing, .checking:
            return // onReady pumps again
        default:
            // Not running, crashed, or the other engine: bring up the right one; onReady pumps.
            backend.ensureWorker(for: kind)
            return
        }
        queue.removeFirst()
        guard let location = installed[job.model.id] else {
            alert = AppAlert(title: "Model not found", message: "\(job.model.name) is no longer on disk.")
            pump()
            return
        }
        activeJob = job
        job.phase = .starting
        job.startedAt = Date()
        // Prompts of the images queued behind this one, for the same model and memory mode, so
        // the worker can encode them while the text encoder is in memory.
        var upcoming: [String] = []
        for queued in queue where queued.model.id == job.model.id && queued.request.lowMemory == job.request.lowMemory {
            let prompt = queued.request.prompt
            if prompt != job.request.prompt, !upcoming.contains(prompt) { upcoming.append(prompt) }
        }
        do {
            try backend.send(.generate(
                jobID: job.id,
                model: job.model,
                modelPath: location.path,
                request: job.request,
                output: job.outputURL,
                upcomingPrompts: Array(upcoming.prefix(8))
            ))
        } catch {
            activeJob = nil
            alert = AppAlert(title: "Couldn’t start generating", message: error.localizedDescription, offersLog: true)
        }
    }

    private func handle(_ event: WorkerEvent) {
        guard let job = activeJob, event.id == nil || event.id == job.id.uuidString else { return }
        switch event.event {
        case "phase":
            switch event.phase {
            case "loading": job.phase = .loadingModel
            case "encoding": job.phase = .encodingPrompt
            case "denoising": job.beginDenoising()
            case "decoding": job.phase = .decoding
            case "encoding_video": job.phase = .encodingVideo
            case "saving": job.phase = .saving
            default: break
            }
        case "progress":
            job.advance(to: event.step ?? job.step)
        case "done":
            complete(job, with: event)
        case "cancelled":
            activeJob = nil
            pump()
        case "failed":
            activeJob = nil
            let dropped = queue.count
            queue.removeAll()
            alert = AppAlert(
                title: "Generation failed",
                message: (event.message ?? "Unknown error")
                    + (dropped > 0 ? "\n\nThe \(dropped) queued images were cancelled." : ""),
                offersLog: true
            )
            updateDockBadge()
        default:
            break
        }
    }

    private func complete(_ job: GenerationJob, with event: WorkerEvent) {
        var request = job.request
        if let width = event.width, let height = event.height {
            request.size = PixelSize(width: width, height: height)
        }
        history.add(HistoryItem(
            id: UUID(),
            createdAt: Date(),
            fileName: job.outputURL.lastPathComponent,
            media: job.model.family.media,
            posterFileName: event.poster.map { URL(fileURLWithPath: $0).lastPathComponent },
            modelID: job.model.id,
            modelName: job.model.name,
            request: request,
            seconds: event.seconds ?? job.startedAt.map { Date().timeIntervalSince($0) } ?? 0,
            peakMemory: event.peakMemory,
            timings: event.timings
        ))
        activeJob = nil
        if queue.isEmpty, !NSApp.isActive {
            NSApp.requestUserAttention(.informationalRequest)
        }
        pump()
    }

    private func backendCrashed(_ message: String) {
        guard isBusy else { return }
        activeJob = nil
        queue.removeAll()
        updateDockBadge()
        alert = AppAlert(title: "The image engine stopped", message: message, offersLog: true)
    }

    private func updateDockBadge() {
        let count = pendingJobs.count
        NSApp?.dockTile.badgeLabel = count > 0 ? "\(count)" : nil
    }

    // MARK: History

    var displayedItem: HistoryItem? {
        switch viewer {
        case .item(let id): history.item(withID: id) ?? history.items.first
        case .live: history.items.first
        }
    }

    /// The live viewer shows the running job instead of the last image.
    var showsLiveJob: Bool { viewer == .live && activeJob != nil }

    func select(_ item: HistoryItem) {
        viewer = item.id == history.items.first?.id && activeJob == nil ? .live : .item(item.id)
    }

    func isSelected(_ item: HistoryItem) -> Bool {
        !showsLiveJob && displayedItem?.id == item.id
    }

    /// Moves through the history; `offset` -1 is newer, +1 is older.
    func moveSelection(by offset: Int) {
        let items = history.items
        guard !items.isEmpty else { return }
        let current = displayedItem.flatMap { item in items.firstIndex { $0.id == item.id } } ?? 0
        let next = min(max(current + offset, 0), items.count - 1)
        select(items[next])
    }

    func delete(_ items: [HistoryItem]) {
        let ids = Set(items.map(\.id))
        if case .item(let id) = viewer, ids.contains(id) { viewer = .live }
        history.remove(ids)
    }

    func clearHistory() {
        viewer = .live
        history.removeAll()
    }

    /// Loads an image's settings back into the controls.
    func reuse(_ item: HistoryItem) {
        // The model first: switching family resets steps and guidance to its defaults.
        if models.contains(where: { $0.id == item.modelID }) { selectedModelID = item.modelID }
        var updated = settings
        if let blocks = item.request.blocks, !blocks.isEmpty {
            updated.blocks = blocks
        } else {
            updated.blocks = [PromptBlock(name: "Prompt 1", text: item.prompt)]
        }
        updated.apply(size: item.size)
        updated.steps = item.request.steps
        updated.guidance = item.request.guidance
        updated.randomSeed = false
        updated.seed = item.request.seed
        updated.transparentBackground = item.request.transparentBackground
        settings = updated
    }

    func url(for item: HistoryItem) -> URL { history.url(for: item) }
    func posterURL(for item: HistoryItem) -> URL { history.posterURL(for: item) }
}
