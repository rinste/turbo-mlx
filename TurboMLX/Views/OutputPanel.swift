import AVFoundation
import AppKit
import SwiftUI

/// Right column: the current output on top, the history below.
struct OutputPanel: View {
    @Environment(AppModel.self) private var app
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.undoManager) private var undoManager
    /// The clip on the stage, here so that space reaches it; the song likewise.
    @State private var clip = ClipPlayer()
    @State private var song = SongPlayer()
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
        // Space plays or pauses a clip or a song, and does nothing else: no Quick Look on an image,
        // and no beep for a key nobody took.
        .onKeyPress(.space) {
            if app.displayedItem?.kind == .video { clip.togglePlayback() }
            if app.displayedItem?.kind == .audio { song.toggle() }
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
        } else if let item = app.displayedItem, item.kind == .model3D {
            // Turned with the mouse or the trackpad, a new stage for each object.
            ObjectStage(url: app.url(for: item))
                .id(item.id)
        } else if let item = app.displayedItem, item.kind == .audio {
            // Its waveform and a play button, a new stage for each song.
            SongStage(player: song, url: app.url(for: item)) { isFocused = true }
                .id(item.id)
        } else if let item = app.displayedItem {
            // A new stage for each image, so it starts fitted.
            ImageStage(item: item) { isFocused = true }
                .id(item.id)
        } else if let model = app.selectedModel, !app.isInstalled(model) {
            ModelDownloadCard(model: model)
        } else {
            let media = app.selectedModel?.family.media ?? .image
            let effects = media == .audio && app.selectedModel?.family.isSong == false
            ContentUnavailableView {
                Label(media == .video ? "No Clips Yet" : media == .model3D ? "No Objects Yet" : effects ? "No Sounds Yet"
                      : media == .audio ? "No Songs Yet" : "No Images Yet",
                      systemImage: media == .video ? "film.stack" : media == .model3D ? "cube" : effects ? "waveform"
                      : media == .audio ? "music.note.list" : "photo.on.rectangle.angled")
            } description: {
                Text(media == .model3D
                     ? "Choose a picture of one object on the left and click Generate.\nThe objects you create stay here, in the history."
                     : effects
                     ? "Describe the sound on the left and click Generate.\nThe sounds you create stay here, in the history."
                     : media == .audio
                     ? "Describe the music and write its lyrics on the left, then click Generate.\nThe songs you create stay here, in the history."
                     : "Write a prompt on the left and click Generate.\nThe \(media == .video ? "clips" : "images") you create stay here, in the history.")
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
            .help("Copy the \(item?.noun ?? MediaKind.image.noun) (⇧⌘C)")
            .disabled(item == nil)

            if let item {
                ShareLink(item: app.url(for: item)) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .help("Share or save the \(item.noun)")
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
            .help("Move the \(item?.noun ?? MediaKind.image.noun) to the Trash")
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
                    VStack(alignment: .leading, spacing: 4) {
                        Text(item.caption)
                        if let lyrics = item.request.lyrics {
                            Text(lyrics)
                                .foregroundStyle(.secondary)
                        }
                        if let negativePrompt = item.request.negativePrompt {
                            Text("Negative: \(negativePrompt)")
                                .foregroundStyle(.secondary)
                        }
                        if let loras = item.request.loras, !loras.isEmpty {
                            Text("LoRA: " + loras.map { "\($0.name) \($0.scaleLabel)" }.joined(separator: ", "))
                                .foregroundStyle(.secondary)
                        }
                    }
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
                    if let gaussians = item.request.numGaussians {
                        MetadataChip(systemImage: "cube", text: "\(ObjectDetail.label(gaussians)) Gaussians",
                                     help: "\(gaussians) Gaussians, from a \(item.request.objectMode == ObjectMode.photo.rawValue ? "photo" : "render")")
                    } else if item.kind == .audio, item.request.lyrics == nil {
                        // A sound effect: its length, to the tenth.
                        MetadataChip(systemImage: "waveform", text: Format.soundDuration(item.request.duration ?? 0))
                    } else if item.kind == .audio {
                        // The length it came out at, 25 frames a second, and the longest asked.
                        let length = item.request.frames.map { Double($0) / 25 } ?? item.request.duration ?? 0
                        MetadataChip(systemImage: "waveform", text: Format.songDuration(length),
                                     help: item.request.duration.map { "Asked for up to \(Format.songDuration($0))" } ?? "")
                    } else {
                        MetadataChip(systemImage: "aspectratio", text: "\(item.size.width)×\(item.size.height)")
                    }
                    if item.kind == .video, let frames = item.request.frames, let fps = item.request.fps, fps > 0 {
                        MetadataChip(systemImage: "film", text: "\(Format.clipDuration(frames: frames, fps: fps)) · \(fps) fps",
                                     help: "\(frames) frames at \(fps) fps")
                    }
                    if item.request.upscale == nil {
                        if item.request.referenceImage != nil, let strength = item.request.imageStrength {
                            // Image to image: how much of the picture could change, as it was asked.
                            MetadataChip(systemImage: "photo", text: "From an image, \(Int(((1 - strength) * 100).rounded()))% changed",
                                         help: "Started from a reference image, keeping \(Int((strength * 100).rounded()))% of it (image strength \(strength.formatted()))")
                        } else if item.request.referenceImage != nil {
                            MetadataChip(systemImage: "photo", text: "From an image",
                                         help: item.kind == .video ? "The clip started from a reference image" : "Made from a reference image")
                        }
                        if item.request.lastFrameImage != nil {
                            MetadataChip(systemImage: "photo", text: "To an image", help: "The clip ended on a reference image")
                        }
                        if item.request.quality == true {
                            MetadataChip(systemImage: "sparkles", text: "Quality",
                                         help: "The full model with guidance (CFG \(Format.guidance(item.request.guidance))) and a negative prompt")
                        }
                        MetadataChip(systemImage: "stairs", text: "\(item.request.steps) steps")
                        if item.kind != .video, item.kind != .audio || item.request.lyrics == nil {
                            MetadataChip(systemImage: "dial.medium", text: "CFG \(Format.guidance(item.request.guidance))")
                        }
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
            ("load", "Model load"), ("encode", item.request.upscale == nil ? "Prompt" : "Picture"), ("compose", "Composing"),
            ("denoise", "Steps"), ("decode", "Decode"), ("save", "Save"),
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
    @Environment(\.undoManager) private var undoManager
    let item: HistoryItem

    var body: some View {
        if let selected = app.selectedModel, selected.family.takesReferenceImage, !selected.family.isUpscaler, item.kind != .model3D,
           item.kind != .audio {
            Button(selected.family.takesLastFrame ? "Use as First Frame"
                   : app.settings.startsFromImage(for: selected.family) ? "Use as Starting Image" : "Use as Reference Image") {
                app.useAsReference(item)
            }
            if selected.family.takesLastFrame {
                Button("Use as Last Frame") { app.useAsReference(item, lastFrame: true) }
            }
        }
        if item.kind == .image, app.upscaler != nil {
            Button("Upscale…") { app.upscale(item, undoManager: undoManager) }
        }
        if !item.prompt.isEmpty {
            Button("Copy Prompt") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(item.prompt, forType: .string)
            }
        }
        Button("Copy \(item.title)") { NSPasteboard.general.copyItem(item, at: app.url(for: item)) }
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
    /// The model picks the clip's length, `frames` being the longest.
    var isLongest = false
    /// A job's frame shows the icon's dots, moving while it runs (`isRunning`); the draft's, sparkles.
    var isJob = false
    var isRunning = false
    /// A song: the longest it may last; a sound effect: its length.
    var songSeconds: Double?

    /// "768 × 512", and the duration for a clip ("up to 5 s" while the model has to pick it); for a 3D object or a
    /// sound, what it is.
    private var sizeLabel: String {
        if model.family.media == .model3D { return "3D object" }
        if model.family.media == .audio, !model.family.isSong { return "Sound, \(Format.soundDuration(songSeconds ?? 10))" }
        if model.family.media == .audio { return "Song, up to \(Format.songDuration(songSeconds ?? 60))" }
        let dimensions = "\(size.width) × \(size.height)"
        guard let frames, let fps, fps > 0 else { return dimensions }
        let duration = Format.clipDuration(frames: frames, fps: fps)
        return "\(dimensions) · \(isLongest ? "up to \(duration)" : duration)"
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
                    if isJob {
                        // As large as the frame leaves room for, up to 120 points.
                        DotGrid(isAnimating: isRunning)
                            .aspectRatio(1, contentMode: .fit)
                            .frame(maxWidth: 120, maxHeight: 120)
                    } else {
                        Image(systemName: "sparkles")
                            .font(.system(size: 34, weight: .light))
                    }
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
            .aspectRatio(size.width > 0 && size.height > 0 ? CGSize(width: size.width, height: size.height) : CGSize(width: 1, height: 1),
                         contentMode: .fit)
            .frame(maxWidth: min(CGFloat(size.width), 520), maxHeight: min(CGFloat(size.height), 520))
    }
}

/// The app icon's dots, alive while a generation runs: the bright middle drifts around the grid and
/// swells in waves from its center outward. The grid is wider than the icon's and fades out toward
/// its edges, so it has no border to show. Still (queued, stopping, or with Reduce Motion), it is the
/// icon's pattern, which is also where the movement starts.
private struct DotGrid: View {
    let isAnimating: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// When the movement started, so that it starts from the icon.
    @State private var start = Date.now

    var body: some View {
        let moves = isAnimating && !reduceMotion
        // 30 frames a second are smooth for this, and leave the display's rate to the machine
        // that is generating.
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !moves)) { timeline in
            Canvas { context, size in
                Self.draw(in: &context, size: size,
                          elapsed: moves ? max(timeline.date.timeIntervalSince(start), 0) : 0)
            }
        }
        .onChange(of: moves) { if $1 { start = .now } }
        .accessibilityHidden(true)
    }

    /// Dots on a side: the icon has 9, the rest is room to fade out in.
    private static let count = 15
    /// The fade, in cells from the grid's center: whole up to `fadeStart`, gone at `fadeEnd`
    /// (before the edges' middles, at 7), drawn in this many steps of opacity.
    private static let fadeStart = 3.4
    private static let fadeEnd = 7.2
    private static let fadeSteps = 24

    /// The dots at `elapsed` seconds in, in the view's foreground style. Measured on the icon
    /// (icon_1024.png): a 9 × 9 grid whose dots go from 18% of the spacing at the corners to 92% in
    /// the middle, as 0.18 + 0.74 · e^−(r / 2.82)^2.85, with r the distance from the middle in cells.
    private static func draw(in context: inout GraphicsContext, size: CGSize, elapsed: Double) {
        let side = min(size.width, size.height)
        let cell = side / CGFloat(count)
        let origin = CGPoint(x: (size.width - side) / 2, y: (size.height - side) / 2)
        let middle = Double(count - 1) / 2
        // The middle wanders on a slow figure of eight, about a cell from the grid's center.
        let wander = 2 * Double.pi * elapsed / 7
        let centerX = middle + 1.3 * sin(wander)
        let centerY = middle + 1.0 * sin(2 * wander)
        // Waves every two seconds, faded in over the first second so the start is the icon.
        let wave = 2 * Double.pi * elapsed / 2
        let swell = 0.12 * min(elapsed, 1)
        // One path for each step of opacity: a fill per dot would be 225 fills a frame.
        var layers = [Path](repeating: Path(), count: fadeSteps)
        for row in 0..<count {
            for column in 0..<count {
                let fromMiddle = hypot(Double(column) - middle, Double(row) - middle)
                let fade = min(max((fromMiddle - fadeStart) / (fadeEnd - fadeStart), 0), 1)
                let opacity = 1 - fade * fade * (3 - 2 * fade)
                let step = Int((opacity * Double(fadeSteps)).rounded(.up))
                guard step > 0 else { continue }
                let distance = hypot(Double(column) - centerX, Double(row) - centerY)
                let reach = 2.82 * (1 + swell * sin(wave - 0.9 * distance))
                let diameter = cell * CGFloat(0.18 + 0.74 * exp(-pow(distance / reach, 2.85)))
                let center = CGPoint(x: origin.x + (CGFloat(column) + 0.5) * cell,
                                     y: origin.y + (CGFloat(row) + 0.5) * cell)
                layers[step - 1].addEllipse(in: CGRect(x: center.x - diameter / 2, y: center.y - diameter / 2,
                                                       width: diameter, height: diameter))
            }
        }
        for (index, layer) in layers.enumerated() where !layer.isEmpty {
            var faded = context
            faded.opacity = Double(index + 1) / Double(fadeSteps)
            faded.fill(layer, with: .foreground)
        }
    }
}

/// The image as it forms: the engine's last preview, small and soft, in the frame the image will
/// have.
private struct PreviewFrame: View {
    let image: NSImage
    let size: PixelSize

    var body: some View {
        Image(nsImage: image)
            .resizable()
            .interpolation(.medium)
            .aspectRatio(CGSize(width: size.width, height: size.height), contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .frame(maxWidth: min(CGFloat(size.width), 520), maxHeight: min(CGFloat(size.height), 520))
            .accessibilityLabel("Preview of the image being generated")
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
                    fps: video ? app.settings.videoFrameRate : nil,
                    isLongest: video && app.settings.autoDuration(for: model.family),
                    songSeconds: model.family.isSong ? app.settings.songSeconds : model.family.media == .audio ? app.settings.soundSeconds : nil
                )
                VStack(spacing: 6) {
                    Text(video ? "New Clip" : model.family.media == .model3D ? "New 3D Object" : model.family.isSong ? "New Song"
                         : model.family.media == .audio ? "New Sound" : "New Image")
                        .font(.headline)
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
            if let preview = job.preview {
                PreviewFrame(image: preview, size: job.request.size)
            } else {
                GenerationFrame(model: job.model, size: job.request.size, frames: job.request.frames, fps: job.request.fps,
                                isLongest: job.framesAreLongest, isJob: true, isRunning: !isQueued && !job.isCancelling,
                                songSeconds: job.request.duration)
            }

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
                    // Twice a second: an upscale's bar follows the time.
                    TimelineView(.periodic(from: .now, by: 0.5)) { context in
                        if let fraction = job.fraction(at: context.date) {
                            ProgressView(value: fraction)
                        } else {
                            ProgressView().progressViewStyle(.linear)
                        }
                    }
                }
                HStack(alignment: .top) {
                    Text(job.request.caption)
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
