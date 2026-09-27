import QuickLook
import SwiftUI

/// Right column: the current output on top, the history below.
struct OutputPanel: View {
    @Environment(AppModel.self) private var app
    @State private var quickLookURL: URL?
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            stage
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            if let item = app.displayedItem, !app.showsLiveJob, showsBackendSetup == false {
                Divider()
                ItemInfoBar(item: item)
            }
            Divider()
            HistoryStrip()
        }
        .background(Color(nsColor: .underPageBackgroundColor))
        .quickLookPreview($quickLookURL)
        .toolbar { toolbarContent }
        .focusable()
        .focused($isFocused)
        .focusEffectDisabled()
        .onMoveCommand { direction in
            switch direction {
            case .left: app.moveSelection(by: -1)
            case .right: app.moveSelection(by: 1)
            default: break
            }
        }
        .onKeyPress(.space) {
            guard let item = app.displayedItem, !app.showsLiveJob else { return .ignored }
            quickLookURL = quickLookURL == nil ? app.url(for: item) : nil
            return .handled
        }
    }

    private var showsBackendSetup: Bool {
        switch app.backend.status {
        case .notInstalled, .installing: true
        case .failed: !app.backend.isInstalled || !app.backend.installOutput.isEmpty
        default: false
        }
    }

    @ViewBuilder
    private var stage: some View {
        if showsBackendSetup {
            BackendSetupView()
        } else if app.showsLiveJob, let job = app.activeJob {
            LiveJobView(job: job)
        } else if let item = app.displayedItem {
            // A new stage for each image, so it starts fitted.
            ImageStage(item: item) { isFocused = true }
                .id(item.id)
        } else if let model = app.selectedModel, !app.isInstalled(model) {
            ModelDownloadCard(model: model)
        } else {
            ContentUnavailableView {
                Label("No Images Yet", systemImage: "photo.on.rectangle.angled")
            } description: {
                Text("Write a prompt on the left and click Generate.\nThe images you create stay here, in the history.")
            }
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .status) {
            BackendStatusButton()
        }
        ToolbarItemGroup(placement: .primaryAction) {
            let item = app.showsLiveJob ? nil : app.displayedItem
            Button {
                if let item { app.reuse(item) }
            } label: {
                Label("Reuse Settings", systemImage: "arrow.uturn.backward.circle")
            }
            .help("Load this image’s prompt, seed and settings into the controls (⌘R)")
            .disabled(item == nil)

            Button {
                if let item { NSPasteboard.general.copyImage(at: app.url(for: item)) }
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .help("Copy the image (⇧⌘C)")
            .disabled(item == nil)

            if let item {
                ShareLink(item: app.url(for: item)) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .help("Share or save the image")
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
            .help("Move the image to the Trash")
            .disabled(item == nil)
        }
    }
}

// MARK: - Image

private struct ImageStage: View {
    @Environment(AppModel.self) private var app
    let item: HistoryItem
    /// A click on the image gives the panel the keyboard (arrows, space).
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

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(item.prompt)
                .font(.callout)
                .lineLimit(3)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(item.prompt)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    MetadataChip(systemImage: "cpu", text: shortModelName, help: "\(item.modelName)\n\(item.modelID)")
                    MetadataChip(systemImage: "aspectratio", text: "\(item.size.width)×\(item.size.height)")
                    MetadataChip(systemImage: "stairs", text: "\(item.request.steps) steps")
                    MetadataChip(systemImage: "dial.medium", text: "CFG \(Format.guidance(item.request.guidance))")
                    MetadataChip(systemImage: "dice", text: "\(item.request.seed)", help: "Seed \(item.request.seed)")
                    MetadataChip(systemImage: "timer", text: Format.duration(item.seconds), help: "Generation time")
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

    /// The family ("Z-Image Turbo") reads better than a catalog name that ends in its memory needs.
    private var shortModelName: String {
        app.models.first { $0.id == item.modelID }?.family.displayName ?? item.modelName
    }
}

/// Actions shared by the viewer and the thumbnails.
struct HistoryItemMenu: View {
    @Environment(AppModel.self) private var app
    let item: HistoryItem

    var body: some View {
        Button("Reuse Settings") { app.reuse(item) }
        Button("Copy Prompt") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(item.prompt, forType: .string)
        }
        Button("Copy Image") { NSPasteboard.general.copyImage(at: app.url(for: item)) }
        Divider()
        Button("Open in Preview") { NSWorkspace.shared.open(app.url(for: item)) }
        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([app.url(for: item)]) }
        Divider()
        Button("Move to Trash", role: .destructive) { app.delete([item]) }
    }
}

// MARK: - Live job

private struct LiveJobView: View {
    @Environment(AppModel.self) private var app
    let job: GenerationJob

    var body: some View {
        let size = job.request.size
        VStack(spacing: 22) {
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
                            .symbolEffect(.pulse, isActive: !job.isCancelling)
                        Text(verbatim: "\(size.width) × \(size.height)")
                            .font(.caption.monospacedDigit())
                    }
                    .foregroundStyle(.secondary)
                }
                .aspectRatio(CGSize(width: size.width, height: size.height), contentMode: .fit)
                .frame(maxWidth: min(CGFloat(size.width), 520), maxHeight: min(CGFloat(size.height), 520))

            VStack(spacing: 10) {
                HStack {
                    Text(job.statusLabel).font(.headline)
                    Spacer()
                    if let remaining = job.estimatedSecondsRemaining {
                        Text(Format.remaining(remaining))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                }
                if let fraction = job.fraction {
                    ProgressView(value: fraction)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
                HStack(alignment: .top) {
                    Text(job.request.prompt)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    Spacer(minLength: 16)
                    Button(job.isCancelling ? "Force Stop" : "Stop", role: .cancel) {
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
}
