import AVFoundation
import SwiftUI

/// Right column: the current output on top, the history below.
struct OutputPanel: View {
    @Environment(AppModel.self) private var app
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.undoManager) private var undoManager
    /// The clip on the stage, here so that space reaches it.
    @State private var clip = ClipPlayer()
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            stage
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            if let item = app.displayedItem {
                Divider()
                ItemInfoBar(item: item)
            }
            Divider()
            HistoryStrip()
        }
        .background(stageBackground)
        // No band behind the toolbar: without a scroll view under it (a clip, the draft, a running
        // job) it would otherwise draw one.
        .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
        .toolbar { toolbarContent }
        .focusable()
        .focused($isFocused)
        .focusEffectDisabled()
        // Left is older, as in the strip.
        .onMoveCommand { direction in
            switch direction {
            case .left: app.moveSelection(by: 1, undoManager: undoManager)
            case .right: app.moveSelection(by: -1, undoManager: undoManager)
            default: break
            }
        }
        // Space plays or pauses a clip, and does nothing else: no Quick Look on an image, and no
        // beep for a key nobody took.
        .onKeyPress(.space) {
            if app.displayedItem?.kind == .video { clip.togglePlayback() }
            return .handled
        }
    }

    /// Darker than the window in both appearances, so the picture stands out and its colors are
    /// judged against a neutral field.
    private var stageBackground: Color {
        colorScheme == .dark ? Color(white: 0.1) : Color(white: 0.82)
    }

    @ViewBuilder
    private var stage: some View {
        if let job = app.displayedJob {
            JobView(job: job)
        } else if app.showsDraft {
            DraftView()
        } else if let item = app.displayedItem, item.kind == .video {
            VideoStage(player: clip, url: app.url(for: item), size: item.size) { isFocused = true }
        } else if let item = app.displayedItem {
            // A new stage for each image, so it starts fitted.
            ImageStage(item: item) { isFocused = true }
                .id(item.id)
        } else if let model = app.selectedModel, !app.isInstalled(model) {
            ModelDownloadCard(model: model)
        } else {
            let video = app.selectedModel?.family.media == .video
            ContentUnavailableView {
                Label(video ? "No Clips Yet" : "No Images Yet", systemImage: video ? "film.stack" : "photo.on.rectangle.angled")
            } description: {
                Text("Write a prompt on the left and click Generate.\nThe \(video ? "clips" : "images") you create stay here, in the history.")
            }
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .status) {
            BackendStatusButton()
        }
        ToolbarItemGroup(placement: .primaryAction) {
            let item = app.displayedItem
            Button {
                if let item { NSPasteboard.general.copyItem(item, at: app.url(for: item)) }
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .help(item?.kind == .video ? "Copy the video (⇧⌘C)" : "Copy the image (⇧⌘C)")
            .disabled(item == nil)

            if let item {
                ShareLink(item: app.url(for: item)) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .help(item.kind == .video ? "Share or save the video" : "Share or save the image")
            }

            Button {
                if let item { NSWorkspace.shared.activateFileViewerSelecting([app.url(for: item)]) }
            } label: {
                Label("Show in Finder", systemImage: "folder")
            }
            .help("Show the file in Finder")
            .disabled(item == nil)

            Button {
                if let item { app.delete([item]) }
            } label: {
                Label("Move to Trash", systemImage: "trash")
            }
            .help(item?.kind == .video ? "Move the video to the Trash" : "Move the image to the Trash")
            .disabled(item == nil)
        }
    }
}

// MARK: - Video

/// A clip from the history, its controls under the picture: AVKit's own lay a dark veil over it
/// while they show. It plays over and over from the start; a click on the picture, like space,
/// pauses or plays it.
private struct VideoStage: View {
    let player: ClipPlayer
    let url: URL
    let size: PixelSize
    /// A click on the picture also gives the panel the keyboard (space, arrows).
    let onClick: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            PlayerSurface(player: player.player)
                .aspectRatio(CGSize(width: size.width, height: size.height), contentMode: .fit)
                .contentShape(Rectangle())
                .onTapGesture {
                    onClick()
                    player.togglePlayback()
                }
            TransportBar(player: player)
                .frame(maxWidth: 560)
        }
        .padding(28)
        .onAppear { player.load(url) }
        .onChange(of: url) { _, url in player.load(url) }
        .onDisappear { player.stop() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            player.applyPlayback()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in
            player.applyPlayback()
        }
    }
}

/// The clip on the stage and what the controls show of it. Each clip plays in a loop, without a gap,
/// until it is paused, and only while Turbo MLX is in front: a clip that finishes, or keeps looping,
/// behind another app would play its sound there. The sound stays as last set, for every clip.
@Observable
private final class ClipPlayer {
    let player = AVQueuePlayer()
    /// Paused by the user; a clip starts playing when it is shown.
    private(set) var isPaused = false
    private(set) var duration: Double = 0
    private(set) var currentTime: Double = 0
    var isMuted = UserDefaults.standard.bool(forKey: ClipPlayer.mutedKey) {
        didSet {
            player.isMuted = isMuted
            UserDefaults.standard.set(isMuted, forKey: Self.mutedKey)
        }
    }

    private static let mutedKey = "mutesClips"
    @ObservationIgnored private var url: URL?
    @ObservationIgnored private var looper: AVPlayerLooper?
    @ObservationIgnored private var timeObserver: Any?
    @ObservationIgnored private var isScrubbing = false

    init() {
        player.isMuted = isMuted
    }

    func load(_ url: URL) {
        stop()
        self.url = url
        isPaused = false
        currentTime = 0
        duration = 0
        let item = AVPlayerItem(url: url)
        looper = AVPlayerLooper(player: player, templateItem: item)
        Task { [weak self] in
            let duration = (try? await item.asset.load(.duration))?.seconds ?? 0
            guard let self, self.url == url else { return }
            self.duration = duration.isFinite ? duration : 0
        }
        timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 30), queue: .main) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self, !self.isScrubbing else { return }
                self.currentTime = time.seconds
            }
        }
        applyPlayback()
    }

    func togglePlayback() {
        isPaused.toggle()
        applyPlayback()
    }

    /// Plays unless paused, while the app is in front; called again when it comes and goes.
    func applyPlayback() {
        guard looper != nil else { return }
        if !isPaused, NSApp.isActive {
            player.play()
        } else {
            player.pause()
        }
    }

    func scrub(to seconds: Double) {
        currentTime = seconds
        player.seek(to: CMTime(seconds: seconds, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func setScrubbing(_ scrubbing: Bool) {
        isScrubbing = scrubbing
    }

    func stop() {
        player.pause()
        looper?.disableLooping()
        looper = nil
        player.removeAllItems()
        if let timeObserver { player.removeTimeObserver(timeObserver) }
        timeObserver = nil
        url = nil
    }
}

/// The picture alone, in a layer. Clicks go through to SwiftUI.
private struct PlayerSurface: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> PlayerLayerView {
        let view = PlayerLayerView()
        view.playerLayer.player = player
        return view
    }

    func updateNSView(_ view: PlayerLayerView, context: Context) {
        if view.playerLayer.player !== player { view.playerLayer.player = player }
    }

    static func dismantleNSView(_ view: PlayerLayerView, coordinator: ()) {
        view.playerLayer.player = nil
    }
}

private final class PlayerLayerView: NSView {
    let playerLayer = AVPlayerLayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        playerLayer.videoGravity = .resizeAspect
        layer = playerLayer
        wantsLayer = true
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

/// Play or pause, the time, a bar to scrub along the clip, and the sound on or off.
private struct TransportBar: View {
    let player: ClipPlayer

    var body: some View {
        HStack(spacing: 12) {
            Button {
                player.togglePlayback()
            } label: {
                Image(systemName: player.isPaused ? "play.fill" : "pause.fill")
                    .frame(width: 18)
            }
            .buttonStyle(.borderless)
            .help(player.isPaused ? "Play (space)" : "Pause (space)")

            Text(verbatim: Self.time(player.currentTime))
                .monospacedDigit()
                .foregroundStyle(.secondary)
            Slider(
                value: Binding(get: { player.currentTime }, set: { player.scrub(to: $0) }),
                in: 0...max(player.duration, 0.01)
            ) { editing in
                player.setScrubbing(editing)
            }
            .controlSize(.small)
            Text(verbatim: Self.time(player.duration))
                .monospacedDigit()
                .foregroundStyle(.secondary)

            Button {
                player.isMuted.toggle()
            } label: {
                Image(systemName: player.isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                    .frame(width: 20)
            }
            .buttonStyle(.borderless)
            .help(player.isMuted ? "Sound on" : "Sound off")
        }
        .font(.callout)
    }

    /// "0:04".
    private static func time(_ seconds: Double) -> String {
        let whole = Int(max(seconds, 0).rounded(.down))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}

// MARK: - Image

private struct ImageStage: View {
    @Environment(AppModel.self) private var app
    let item: HistoryItem
    /// A click on the image gives the panel the keyboard (arrows).
    let onClick: () -> Void

    @State private var zoom = ImageZoom()
    @State private var isHovering = false

    var body: some View {
        ZoomableImage(
            url: app.url(for: item),
            size: item.size,
            showsCheckerboard: item.request.transparentBackground,
            zoom: zoom,
            onMouseDown: onClick
        )
        .contextMenu { HistoryItemMenu(item: item) }
        .overlay(alignment: .bottomTrailing) {
            let showsControls = isHovering || zoom.isZoomedIn
            ZoomControls(zoom: zoom)
                .padding(12)
                .opacity(showsControls ? 1 : 0)
                .allowsHitTesting(showsControls)
                .animation(.easeOut(duration: 0.15), value: showsControls)
        }
        .onHover { isHovering = $0 }
        .focusedSceneValue(\.imageZoom, zoom)
    }
}

private struct ItemInfoBar: View {
    @Environment(AppModel.self) private var app
    let item: HistoryItem

    /// Five lines of callout text, 15 pt each: a longer prompt scrolls, so the image keeps its room.
    private static let promptHeight: CGFloat = 75
    /// The prompt's scroll bar goes in the bar's margin instead of over the text.
    private static let scrollBarMargin: CGFloat = 12

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HeightLimit(maxHeight: Self.promptHeight) {
                ScrollView {
                    Text(item.prompt)
                        .font(.callout)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .contentMargins(.trailing, Self.scrollBarMargin, for: .scrollContent)
                .scrollBounceBehavior(.basedOnSize)
                // Shows that a long prompt goes on below.
                .scrollIndicatorsFlash(onAppear: true)
            }
            .padding(.trailing, -Self.scrollBarMargin)
            // Each image's prompt starts from the top.
            .id(item.id)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    MetadataChip(systemImage: "cpu", text: shortModelName, help: "\(item.modelName)\n\(item.modelID)")
                    MetadataChip(systemImage: "aspectratio", text: "\(item.size.width)×\(item.size.height)")
                    if item.kind == .video, let frames = item.request.frames, let fps = item.request.fps, fps > 0 {
                        MetadataChip(systemImage: "film", text: "\(Format.clipDuration(frames: frames, fps: fps)) · \(fps) fps",
                                     help: "\(frames) frames at \(fps) fps")
                    }
                    if item.request.referenceImage != nil {
                        MetadataChip(systemImage: "photo", text: "From an image",
                                     help: item.kind == .video ? "The clip started from a reference image" : "Made from a reference image")
                    }
                    MetadataChip(systemImage: "stairs", text: "\(item.request.steps) steps")
                    if item.kind == .image {
                        MetadataChip(systemImage: "dial.medium", text: "CFG \(Format.guidance(item.request.guidance))")
                    }
                    MetadataChip(systemImage: "dice", text: "\(item.request.seed)", help: "Seed \(item.request.seed)")
                    MetadataChip(systemImage: "timer", text: Format.duration(item.seconds), help: timingHelp)
                    if let peak = item.peakMemory {
                        MetadataChip(systemImage: "memorychip", text: Format.memory(peak), help: "Peak memory")
                    }
                    MetadataChip(
                        systemImage: "calendar",
                        text: item.createdAt.formatted(date: .abbreviated, time: .shortened)
                    )
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.background)
    }

    /// "Generation time" plus, when the engine reported them, the seconds of each phase.
    private var timingHelp: String {
        let phases: [(key: String, label: String)] = [
            ("load", "Model load"), ("encode", "Prompt"), ("denoise", "Steps"), ("decode", "Decode"), ("save", "Save"),
        ]
        guard let timings = item.timings else { return "Generation time" }
        let lines = phases.compactMap { phase in timings[phase.key].map { "\(phase.label): \(Format.seconds($0))" } }
        return (["Generation time"] + lines).joined(separator: "\n")
    }

    /// The family ("Z-Image Turbo") reads better than a catalog name that ends in its memory needs.
    private var shortModelName: String {
        app.models.first { $0.id == item.modelID }?.family.displayName ?? item.modelName
    }
}

/// Its content at the height it needs, up to `maxHeight`: a scroll view in it is as tall as a
/// short text, and scrolls a long one.
private nonisolated struct HeightLimit: Layout {
    let maxHeight: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        guard let content = subviews.first else { return .zero }
        let ideal = content.sizeThatFits(ProposedViewSize(width: proposal.width, height: nil))
        return CGSize(
            width: proposal.width ?? ideal.width,
            height: min(ideal.height, maxHeight, proposal.height ?? .infinity)
        )
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        subviews.first?.place(at: bounds.origin, proposal: ProposedViewSize(bounds.size))
    }
}

/// Actions shared by the viewer and the thumbnails.
struct HistoryItemMenu: View {
    @Environment(AppModel.self) private var app
    let item: HistoryItem

    var body: some View {
        if app.selectedModel?.family.takesReferenceImage == true {
            Button("Use as Reference Image") { app.useAsReference(item) }
        }
        Button("Copy Prompt") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(item.prompt, forType: .string)
        }
        Button(item.kind == .video ? "Copy Video" : "Copy Image") { NSPasteboard.general.copyItem(item, at: app.url(for: item)) }
        Divider()
        let url = app.url(for: item)
        Button(Self.openTitle(for: url)) { NSWorkspace.shared.open(url) }
        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([app.url(for: item)]) }
        Divider()
        Button("Move to Trash", role: .destructive) { app.delete([item]) }
    }

    /// "Open in Preview" for an image, "Open in QuickTime Player" for a clip: the app that opens it.
    private static func openTitle(for url: URL) -> String {
        guard let application = NSWorkspace.shared.urlForApplication(toOpen: url), let bundle = Bundle(url: application),
              let name = (bundle.localizedInfoDictionary?["CFBundleDisplayName"] ?? bundle.infoDictionary?["CFBundleDisplayName"]
                          ?? bundle.infoDictionary?["CFBundleName"]) as? String
        else { return "Open" }
        return "Open in \(name)"
    }
}

// MARK: - Live job

/// The outline a generation fills, in its shape: which model, and how big.
private struct GenerationFrame: View {
    let model: ModelDescriptor
    let size: PixelSize
    let frames: Int?
    let fps: Int?
    var isPulsing = false

    /// "768 × 512", and the duration for a clip.
    private var sizeLabel: String {
        let dimensions = "\(size.width) × \(size.height)"
        guard let frames, let fps, fps > 0 else { return dimensions }
        return "\(dimensions) · \(Format.clipDuration(frames: frames, fps: fps))"
    }

    var body: some View {
        RoundedRectangle(cornerRadius: 6)
            .fill(.quaternary.opacity(0.5))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(.tertiary, style: StrokeStyle(lineWidth: 1, dash: [6, 5]))
            }
            .overlay {
                VStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 34, weight: .light))
                        .symbolEffect(.pulse, isActive: isPulsing)
                    // The family, as under a finished image; the tooltip has the full name.
                    Text(model.family.displayName)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                        .help(model.name)
                    Text(verbatim: sizeLabel)
                        .font(.caption.monospacedDigit())
                }
                .foregroundStyle(.secondary)
                .padding(8)
            }
            .aspectRatio(CGSize(width: size.width, height: size.height), contentMode: .fit)
            .frame(maxWidth: min(CGFloat(size.width), 520), maxHeight: min(CGFloat(size.height), 520))
    }
}

/// The "+" after the history: the frame of the next generation as the controls on the left set
/// it up, until Generate starts it.
private struct DraftView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        if let model = app.selectedModel {
            let video = model.family.media == .video
            VStack(spacing: 22) {
                GenerationFrame(
                    model: model,
                    size: app.settings.size(for: model.family),
                    frames: video ? app.settings.videoFrames : nil,
                    fps: video ? app.settings.videoFrameRate : nil
                )
                VStack(spacing: 6) {
                    Text(video ? "New Clip" : "New Image").font(.headline)
                    Text("Set it up on the left, then Generate (⌘↩). What you change here stays for the next time you come back to +.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: 520)
            }
            .padding(28)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

/// A job running, with its progress, or waiting in the queue, with when it starts.
private struct JobView: View {
    @Environment(AppModel.self) private var app
    let job: GenerationJob

    var body: some View {
        let isQueued = job.phase == .queued
        VStack(spacing: 22) {
            GenerationFrame(model: job.model, size: job.request.size, frames: job.request.frames, fps: job.request.fps,
                            isPulsing: !isQueued && !job.isCancelling)

            VStack(spacing: 10) {
                HStack {
                    Text(job.statusLabel).font(.headline)
                    Spacer()
                    Group {
                        if isQueued {
                            Text(turn)
                        } else {
                            TimeLeft(job: job)
                        }
                    }
                    .foregroundStyle(.secondary)
                }
                if !isQueued {
                    if let fraction = job.fraction {
                        ProgressView(value: fraction)
                    } else {
                        ProgressView().progressViewStyle(.linear)
                    }
                }
                HStack(alignment: .top) {
                    Text(job.request.prompt)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    Spacer(minLength: 16)
                    Button(isQueued ? "Remove from Queue" : job.isCancelling ? "Force Stop" : "Stop", role: .cancel) {
                        app.cancel(job)
                    }
                }
                if job.phase == .loadingModel {
                    Text("The first load reads the whole model from disk; later images start right away.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: 520)
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// A queued job's turn, as the Generate button counts it.
    private var turn: String {
        switch app.pendingJobs.firstIndex(where: { $0 === job }) ?? 0 {
        case 0: "Starts when the engine is ready"
        case 1: "Starts when the one in progress is done"
        case let ahead: "Starts after the \(ahead) ahead of it"
        }
    }
}
