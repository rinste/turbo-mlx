import AppKit
import Foundation
import Observation

struct AppAlert: Identifiable {
    let id = UUID()
    var title: String
    var message: String
    var offersLog = false
    /// A button that opens a web page, such as a new release's.
    var link: (title: String, url: URL)?

    static func update(_ release: UpdateChecker.Release) -> AppAlert {
        AppAlert(
            title: "Turbo MLX \(release.version) is available",
            message: "You have version \(UpdateChecker.currentVersion). Download the new one from its release page and put it in Applications in place of this one: images, settings and models stay.",
            link: ("Download", release.page)
        )
    }
}

/// App-wide state: model catalog, generation settings, the job queue and the history.
@Observable
final class AppModel {
    /// What the image viewer shows: the job in progress / latest result, a picked history item, or
    /// the frame of the next generation (the "+" after the history).
    enum ViewerSelection: Hashable {
        case live
        case item(HistoryItem.ID)
        case draft
    }

    /// Why "Generate" is not available right now.
    enum Blocker: Equatable {
        case noModel
        case modelNotDownloaded
        case modelDownloading
        case emptyPrompt

        var hint: String {
            switch self {
            case .noModel: "Choose a model."
            case .modelNotDownloaded: "Download the model to get started."
            case .modelDownloading: "Waiting for the download to finish."
            case .emptyPrompt: "Write a prompt."
            }
        }
    }

    let backend = BackendController()
    let downloads = DownloadCenter()
    let history = HistoryStore()
    let updates = UpdateChecker()

    private(set) var models: [ModelDescriptor]
    private(set) var installed: [String: URL] = [:]
    /// Bytes each Hugging Face repository of the list (companions too) takes in the cache, partial
    /// downloads included: what moving a model to the Trash frees. Measured with `installed`.
    private(set) var bytesOnDisk: [String: Int64] = [:]
    let locator = ModelLocator()

    var selectedModelID: String {
        didSet {
            UserDefaults.standard.set(selectedModelID, forKey: Keys.selectedModel)
            if controlsSource == .draft { storeDraft() }
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
            if controlsSource == .draft { storeDraft() }
            preloadIfUseful()
        }
    }

    /// Where the controls' settings come from: the draft behind the "+", which keeps every change
    /// made to it, or a history item being looked at, whose settings are only shown.
    enum ControlsSource: Equatable {
        case draft
        case item
    }

    private(set) var controlsSource = ControlsSource.draft

    /// The settings and model the "+" brings back, as last changed there (saved across launches).
    @ObservationIgnored private var draft: (settings: GenerationSettings, modelID: String)?

    private func storeDraft() {
        draft = (settings, selectedModelID)
        let defaults = UserDefaults.standard
        if let data = try? JSONEncoder().encode(settings) { defaults.set(data, forKey: Keys.draftSettings) }
        defaults.set(selectedModelID, forKey: Keys.draftModel)
    }

    /// Model path the worker was asked to load ahead of time, so the request is not repeated.
    private var preloadRequested: String?
    /// Set when the user frees the memory: no preloading until the model changes or an image runs.
    @ObservationIgnored private var preloadDeclined = false
    @ObservationIgnored private var preloadScheduled = false

    private(set) var queue: [GenerationJob] = []
    private(set) var activeJob: GenerationJob?
    var viewer = ViewerSelection.live
    var alert: AppAlert?
    /// Model to generate with as soon as its download completes ("Download and Generate").
    private(set) var generateAfterDownload: String?
    @ObservationIgnored private var started = false
    /// A queue left running when the user walks away finishes instead of waiting for the Mac to wake.
    @ObservationIgnored private let awake = KeepAwake(reason: "Generating images and videos")

    private enum Keys {
        static let settings = "generationSettings"
        static let selectedModel = "selectedModelID"
        static let customModels = "customModels"
        static let draftSettings = "draftSettings"
        static let draftModel = "draftModelID"
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
            settings = Self.initialSettings(for: catalog.first { $0.id == selectedID })
        }
        // The controls open on the draft, as the "+" left it.
        if let data = defaults.data(forKey: Keys.draftSettings),
           let saved = try? JSONDecoder().decode(GenerationSettings.self, from: data),
           let modelID = defaults.string(forKey: Keys.draftModel), catalog.contains(where: { $0.id == modelID }) {
            settings = saved
            selectedModelID = modelID
        }
        draft = (settings, selectedModelID)

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
        history.pruneReferences(keeping: settings.referenceImage)
        // Local model folders before the engine, which inherits the access they open.
        FolderAccess.restore()
        refreshInstalled()
        backend.ensureWorker()
        checkForUpdatesInBackground()
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

    /// The model the engine holds in memory, if it is one of the catalog's.
    var loadedModel: ModelDescriptor? {
        guard let path = backend.loadedModelPath else { return nil }
        return models.first { installed[$0.id]?.path == path }
    }

    /// The model being loaded ahead of time, until it is in memory (or the load fails).
    var loadingModel: ModelDescriptor? {
        guard let path = preloadRequested, path != backend.loadedModelPath else { return nil }
        return models.first { installed[$0.id]?.path == path }
    }

    func isLoaded(_ model: ModelDescriptor) -> Bool {
        guard let path = backend.loadedModelPath else { return false }
        return installed[model.id]?.path == path
    }

    func refreshInstalled() {
        var found: [String: URL] = [:]
        var bytes: [String: Int64] = [:]
        for model in models {
            if let url = locator.installedLocation(of: model) { found[model.id] = url }
            for repo in [model.repo, model.family.companion?.repo].compactMap(\.self) where bytes[repo] == nil {
                bytes[repo] = locator.bytesOnDisk(of: repo)
            }
        }
        installed = found
        bytesOnDisk = bytes
    }

    /// Downloads the model (checking the disk has room for what is still missing first).
    func download(_ model: ModelDescriptor) {
        downloads.start(model, locator: locator)
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
        // What arrived stays, to resume from or to move to the Trash.
        refreshInstalled()
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
    /// busy, when the model is already in memory, or after "Free Memory". An engine that is not
    /// running (stopped, or crashed) is brought up first.
    ///
    /// At most once every 1.5 s, for the model selected by then: clicking through the history
    /// switches models, and each switch would otherwise start loading one.
    private func preloadIfUseful() {
        guard !preloadScheduled else { return }
        preloadScheduled = true
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(1.5))
            self?.preloadScheduled = false
            self?.preloadNow()
        }
    }

    private func preloadNow() {
        guard !preloadDeclined, activeJob == nil, queue.isEmpty, !settings.trimmedPrompt.isEmpty,
              let model = selectedModel, let location = installed[model.id]
        else { return }
        if !backend.isRunning {
            switch backend.status {
            case .stopped, .failed: backend.ensureWorker() // onReady comes back here
            default: break
            }
            return
        }
        guard backend.status == .ready, backend.loadedModelPath != location.path, preloadRequested != location.path
        else { return }
        preloadRequested = location.path
        try? backend.send(.load(model: model, modelPath: location.path, textEncoderPath: locator.companionLocation(of: model)?.path,
                                lowMemory: settings.lowMemory))
    }

    /// What moving a model to the Trash takes away: its repository in the Hugging Face cache and,
    /// when no other downloaded model needs it, its companion. Nil for a local folder, or when
    /// nothing of the model is on disk (a download stopped halfway counts).
    func downloadFiles(of model: ModelDescriptor) -> (repos: [String], bytes: Int64)? {
        guard let repo = model.repo, bytesOnDisk[repo, default: 0] > 0 else { return nil }
        var repos = [repo]
        if let companion = model.family.companion,
           !models.contains(where: { $0.id != model.id && $0.family.companion?.repo == companion.repo && isInstalled($0) }) {
            repos.append(companion.repo)
        }
        return (repos, repos.reduce(0) { $0 + bytesOnDisk[$1, default: 0] })
    }

    /// Not while the model is downloading or has images waiting to be generated.
    func canTrash(_ model: ModelDescriptor) -> Bool {
        !downloads.isDownloading(model) && !pendingJobs.contains { $0.model.id == model.id } && downloadFiles(of: model) != nil
    }

    /// Moves the model's files to the Trash, where they can still be put back until it is
    /// emptied; the model stays in the list, to be downloaded again.
    func trashDownload(of model: ModelDescriptor) {
        guard canTrash(model), let files = downloadFiles(of: model) else { return }
        if isLoaded(model) || loadingModel?.id == model.id {
            preloadRequested = nil
            backend.unloadModel()
        }
        var failure: String?
        for repo in files.repos {
            do {
                try FileManager.default.trashItem(at: locator.repoFolder(repo), resultingItemURL: nil)
            } catch {
                failure = failure ?? error.localizedDescription
            }
        }
        downloads.clearFailure(model)
        refreshInstalled()
        if let failure {
            alert = AppAlert(title: "Couldn’t move \(model.name) to the Trash", message: failure)
        }
    }

    // MARK: Updates

    /// Turbo MLX → Check for Updates: says what it found, "up to date" and failures included.
    func checkForUpdates() {
        Task {
            do {
                if let release = try await updates.check() {
                    alert = .update(release)
                } else {
                    alert = AppAlert(title: "Turbo MLX is up to date",
                                     message: "Version \(UpdateChecker.currentVersion) is the latest.")
                }
            } catch {
                alert = AppAlert(title: "Couldn’t check for updates", message: error.localizedDescription)
            }
        }
    }

    /// At launch, at most once a day: a newer version is announced once, and not over another
    /// alert; a failed check stays quiet.
    private func checkForUpdatesInBackground() {
        guard updates.isDue else { return }
        Task {
            guard let release = try? await updates.check(), alert == nil, updates.announcesOnce(release) else { return }
            alert = .update(release)
        }
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
        // A folder picked just now: the running engine started before its access was opened.
        // Restarted first, so the model is preloaded once, when the new engine is ready.
        if case .local = model.source, backend.isRunning, !isBusy { backend.restartWorker() }
        selectedModelID = model.id
    }

    func removeCustomModel(_ model: ModelDescriptor) {
        guard !model.isBuiltIn else { return }
        cancelDownload(model)
        if case .local(let path) = model.source { FolderAccess.forget(path) }
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
        let isVideo = model.family.media == .video
        // A reference image whose file is gone (a history folder emptied by hand) is dropped.
        let reference = settings.referenceImage.flatMap { name in
            FileManager.default.fileExists(atPath: HistoryStore.referenceURL(name).path) ? name : nil
        }
        for seed in settings.nextSeeds() {
            let request = GenerationRequest(
                prompt: settings.trimmedPrompt,
                blocks: settings.blocks.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty },
                seed: seed,
                size: settings.size(for: model.family),
                steps: min(max(settings.steps, model.stepRange.lowerBound), model.stepRange.upperBound),
                guidance: settings.guidance,
                transparentBackground: settings.transparentBackground && model.family.producesAlpha,
                lowMemory: settings.lowMemory,
                frames: isVideo ? settings.videoFrames : nil,
                fps: isVideo ? settings.videoFrameRate : nil,
                referenceImage: isVideo && model.family.takesReferenceImage ? reference : nil
            )
            let output = isVideo ? history.newVideoURL(seed: seed) : history.newImageURL(seed: seed)
            let job = GenerationJob(model: model, request: request, outputURL: output)
            job.plan = TimeEstimate.plan(model: model, request: request, history: history.items, models: models)
            queue.append(job)
        }
        viewer = .live
        pump()
    }

    /// What the "+" after the history starts from: the last generation queued, running or finished.
    var lastGeneration: (model: ModelDescriptor, request: GenerationRequest)? {
        if let job = queue.last ?? activeJob { return (job.model, job.request) }
        guard let item = history.items.first, let model = models.first(where: { $0.id == item.modelID }) else { return nil }
        return (model, item.request)
    }

    /// The "+": the draft on the left, as last changed there, and the frame of the next generation
    /// on the right, for Generate to start. Without a draft yet, the last generation's settings.
    func startDraft(undoManager: UndoManager? = nil) {
        var target = draft ?? (settings, selectedModelID)
        if draft == nil, let (model, request) = lastGeneration {
            target = controls(from: request, modelID: model.id, media: model.family.media)
        }
        replaceControls(with: target.settings, modelID: target.modelID, viewer: .draft, source: .draft, actionName: "New",
                        undoManager: undoManager)
    }

    var showsDraft: Bool { viewer == .draft }

    func cancel(_ job: GenerationJob) {
        if let index = queue.firstIndex(where: { $0 === job }) {
            queue.remove(at: index)
            queueChanged()
            return
        }
        guard job === activeJob else { return }
        if job.isCancelling {
            // Asked twice: the worker is inside a step that cannot be interrupted (loading,
            // prompt encoding, decoding). Restart it; the model will be reloaded next time.
            activeJob = nil
            backend.restartWorker()
            queueChanged()
        } else {
            job.isCancelling = true
            try? backend.send(.cancel(jobID: job.id))
        }
    }

    func cancelAll() {
        queue.removeAll()
        if let activeJob { cancel(activeJob) }
        queueChanged()
    }

    private func pump() {
        defer { queueChanged() }
        guard activeJob == nil, let job = queue.first else { return }
        switch backend.status {
        case .ready:
            break
        case .starting, .checking:
            return // onReady pumps again
        case .stopped, .failed:
            // Not running, or crashed: bring it up; onReady pumps.
            backend.ensureWorker()
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
                textEncoderPath: locator.companionLocation(of: job.model)?.path,
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
        if event.event == "load_failed" {
            preloadRequested = nil
            return
        }
        guard let job = activeJob, event.id == nil || event.id == job.id.uuidString else { return }
        if let total = event.total, event.event == "progress" || event.phase == "denoising" {
            job.reportedTotal = Int(total)
        }
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
            queueChanged()
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
        queueChanged()
        alert = AppAlert(title: "The engine stopped", message: message, offersLog: true)
    }

    /// The Dock badge counts what is left, and the Mac stays awake until it is done.
    private func queueChanged() {
        let count = pendingJobs.count
        NSApp?.dockTile.badgeLabel = count > 0 ? "\(count)" : nil
        awake.isOn = count > 0
    }

    // MARK: History

    var displayedItem: HistoryItem? {
        switch viewer {
        case .item(let id): history.item(withID: id) ?? history.items.first
        case .live: history.items.first
        case .draft: nil
        }
    }

    /// The live viewer shows the running job instead of the last image.
    var showsLiveJob: Bool { viewer == .live && activeJob != nil }

    /// Shows a history item and puts its prompt, model and settings in the controls on the left.
    /// `undoManager` is the window's, where ⌘Z finds what they replaced.
    func select(_ item: HistoryItem, undoManager: UndoManager? = nil) {
        let controls = controls(from: item.request, modelID: item.modelID, media: item.kind)
        replaceControls(with: controls.settings, modelID: controls.modelID,
                        viewer: item.id == history.items.first?.id && activeJob == nil ? .live : .item(item.id),
                        source: .item, actionName: "Load Settings", undoManager: undoManager)
    }

    func isSelected(_ item: HistoryItem) -> Bool {
        !showsLiveJob && displayedItem?.id == item.id
    }

    /// Moves through the history, loading each item's settings as a click would; `offset` -1 is
    /// newer (to the right in the strip), +1 is older.
    func moveSelection(by offset: Int, undoManager: UndoManager? = nil) {
        let items = history.items
        guard !items.isEmpty else { return }
        // The draft sits after the newest: left of it is the newest, right of it nothing.
        if viewer == .draft {
            if offset > 0 { select(items[0], undoManager: undoManager) }
            return
        }
        let current = displayedItem.flatMap { item in items.firstIndex { $0.id == item.id } } ?? 0
        let next = min(max(current + offset, 0), items.count - 1)
        select(items[next], undoManager: undoManager)
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

    /// A generation's prompt blocks, model and settings, as the controls should show them. Its seed
    /// goes in the seed field with Random on, as always by default: Generate makes a variation, and
    /// turning Random off makes the same one again.
    private func controls(from request: GenerationRequest, modelID: String, media: MediaKind)
        -> (settings: GenerationSettings, modelID: String) {
        let model = models.first { $0.id == modelID }
        var updated = settings
        if let blocks = request.blocks, !blocks.isEmpty {
            updated.blocks = blocks
        } else {
            updated.blocks = PromptBlock.defaults(subject: request.prompt)
        }
        let isVideo = media == .video
        updated.apply(size: request.size, video: isVideo)
        if isVideo, let frames = request.frames, let fps = request.fps, fps > 0 {
            updated.videoFrameRate = fps
            updated.videoSeconds = min(max((Double(frames - 1) / Double(fps)).rounded(), GenerationSettings.videoDurations.lowerBound),
                                       GenerationSettings.videoDurations.upperBound)
        }
        if isVideo {
            updated.referenceImage = request.referenceImage.flatMap { name in
                FileManager.default.fileExists(atPath: HistoryStore.referenceURL(name).path) ? name : nil
            }
        }
        updated.steps = request.steps
        updated.guidance = request.guidance
        updated.seed = request.seed
        updated.randomSeed = true
        // Only a model that makes transparent images says which background was chosen: the
        // others always save false, which would turn the choice off for the next Ming image.
        if model?.family.producesAlpha == true {
            updated.transparentBackground = request.transparentBackground
        }
        return (updated, model?.id ?? selectedModelID)
    }

    /// Reset: the model and settings of a first launch, with an empty prompt, as a new draft.
    func resetControls(undoManager: UndoManager? = nil) {
        let model = models.first { $0.id == ModelCatalog.defaultModelID } ?? models.first
        replaceControls(with: Self.initialSettings(for: model), modelID: model?.id ?? selectedModelID, viewer: .draft,
                        source: .draft, actionName: "Reset", undoManager: undoManager)
    }

    /// Settings for a first launch: the model's steps and guidance, Save memory below 64 GB.
    private static func initialSettings(for model: ModelDescriptor?) -> GenerationSettings {
        var initial = GenerationSettings()
        // At 1024 px with everything resident Ming-Image peaks at ~35 GB and Qwen-Image at
        // ~43 GB (~15 and ~21 GB with Save memory): below 64 GB the full-speed mode would swap.
        initial.lowMemory = ProcessInfo.processInfo.physicalMemory < 64 << 30
        if let model {
            initial.steps = model.defaultSteps
            initial.guidance = model.defaultGuidance
        }
        return initial
    }

    /// Puts other settings and another model in the controls, and shows `newViewer`, so that ⌘Z
    /// brings back what they replaced and what was shown: a click in the history, the "+" or Reset
    /// does not lose a prompt being written. Without the view's undo manager (a menu command), the
    /// main window's, which exists while the app is active.
    private func replaceControls(
        with new: GenerationSettings, modelID: String, viewer newViewer: ViewerSelection? = nil,
        source newSource: ControlsSource? = nil, actionName: String, undoManager: UndoManager?
    ) {
        let previous = (settings: settings, modelID: selectedModelID, viewer: viewer, source: controlsSource)
        let shown = newViewer ?? viewer
        let source = newSource ?? controlsSource
        guard new != previous.settings || modelID != previous.modelID || shown != previous.viewer || source != previous.source
        else { return }
        if let undoManager = undoManager ?? NSApp.mainWindow?.undoManager {
            undoManager.registerUndo(withTarget: self) { model in
                MainActor.assumeIsolated {
                    model.replaceControls(with: previous.settings, modelID: previous.modelID, viewer: previous.viewer,
                                          source: previous.source, actionName: actionName, undoManager: undoManager)
                }
            }
            undoManager.setActionName(actionName)
        }
        // The source first, so that a history item's settings do not land in the draft; then the
        // model: switching family resets steps and guidance, which `new` then sets.
        controlsSource = source
        selectedModelID = modelID
        settings = new
        viewer = shown
    }

    /// Makes a generated image (a clip's first frame) the reference image of the next clip.
    func useAsReference(_ item: HistoryItem) {
        let url = url(for: item)
        Task {
            do {
                settings.referenceImage = try await HistoryStore.importReference(from: url)
            } catch {
                alert = AppAlert(title: "Couldn’t use the image", message: error.localizedDescription)
            }
        }
    }

    func url(for item: HistoryItem) -> URL { history.url(for: item) }
    func posterURL(for item: HistoryItem) -> URL { history.posterURL(for: item) }
}
