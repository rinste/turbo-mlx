import SwiftUI
import UniformTypeIdentifiers

/// Film strip of past images and clips, by date (the newest on the right) or in the order the user
/// dragged them into, then the running and queued jobs and a "+" that sets up the next one. A click
/// shows an item or a job and puts its settings on the left.
struct HistoryStrip: View {
    @Environment(AppModel.self) private var app
    @State private var confirmsClear = false
    /// At the end, where the newest are, until the user scrolls away.
    @State private var position = ScrollPosition(edge: .trailing)

    private let thumbnailHeight: CGFloat = 92

    var body: some View {
        @Bindable var app = app
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text("History").font(.headline)
                Text("\(app.history.items.count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Menu {
                    Picker("Order", selection: $app.historyOrder) {
                        ForEach(HistoryOrder.allCases) { order in
                            Text(order.title).tag(order)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                } label: {
                    Label(app.historyOrder.title, systemImage: "arrow.up.arrow.down")
                        .font(.caption)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .padding(.leading, 6)
                .help("By date, the newest on the right, or in your own order: drag the images and clips where you want them")
                Spacer()
                Menu {
                    Button("Show Folder in Finder") {
                        NSWorkspace.shared.open(HistoryStore.directory)
                    }
                    Button("Clear Queue") { app.cancelAll() }
                        .disabled(!app.isBusy)
                    Divider()
                    Button("Clear History…", role: .destructive) { confirmsClear = true }
                        .disabled(app.history.items.isEmpty)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
            }
            .padding(.horizontal, 14)
            .padding(.top, 8)

            ScrollViewReader { proxy in
                ScrollView(.horizontal) {
                    LazyHStack(spacing: 8) {
                        ForEach(app.arrangedHistory) { item in
                            HistoryThumbnail(item: item, height: thumbnailHeight, isSelected: app.isSelected(item))
                                .id(item.id)
                        }
                        // The running job next to the images it will join, then the queue.
                        ForEach(app.pendingJobs) { job in
                            JobThumbnail(job: job, height: thumbnailHeight, isSelected: app.isSelected(job))
                                .id(job.id)
                        }
                        if app.history.items.isEmpty && app.pendingJobs.isEmpty {
                            Text("Generated images will appear here.")
                                .font(.callout)
                                .foregroundStyle(.tertiary)
                                .frame(height: thumbnailHeight)
                        } else {
                            NewTile(height: thumbnailHeight)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                }
                .scrollIndicators(.automatic)
                .scrollPosition($position)
                .onChange(of: app.displayedItem?.id ?? app.displayedJob?.id) { _, id in
                    guard let id else { return }
                    withAnimation(.snappy) { proxy.scrollTo(id, anchor: .center) }
                }
                // A new job: back to the end, where it is. Not after a deletion, which leaves the
                // strip where it was.
                .onChange(of: app.history.items.count + app.pendingJobs.count) { old, new in
                    guard new > old else { return }
                    withAnimation(.snappy) { position.scrollTo(edge: .trailing) }
                }
            }
        }
        .frame(height: thumbnailHeight + 48)
        .background(.background)
        .confirmationDialog(
            "Clear the history?",
            isPresented: $confirmsClear
        ) {
            Button("Move \(app.history.items.count) Images to the Trash", role: .destructive) {
                app.clearHistory()
            }
        } message: {
            Text("The images go to the Trash, so you can still recover them from Finder.")
        }
    }
}

/// After the last generation: the next one, set up with its settings on the left and its frame on
/// the right, for Generate to start.
private struct NewTile: View {
    @Environment(AppModel.self) private var app
    @Environment(\.undoManager) private var undoManager
    let height: CGFloat
    @State private var isHovered = false

    var body: some View {
        let isSelected = app.showsDraft
        Button { app.startDraft(undoManager: undoManager) } label: {
            RoundedRectangle(cornerRadius: 6)
                .fill(isHovered || isSelected ? AnyShapeStyle(.quaternary) : AnyShapeStyle(.clear))
                .overlay {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 6).strokeBorder(Color.active, lineWidth: 3)
                    } else {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(.tertiary, style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    }
                }
                .overlay {
                    Image(systemName: "plus")
                        .font(.title2.weight(.medium))
                        .foregroundStyle(.secondary)
                }
                .frame(width: height * 0.7, height: height)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help("New: the last one’s settings on the left, ready to change and generate")
        .accessibilityLabel("New")
    }
}

private func thumbnailWidth(for size: PixelSize, height: CGFloat) -> CGFloat {
    let ratio = CGFloat(size.width) / CGFloat(max(size.height, 1))
    return min(max(height * ratio, height * 0.56), height * 1.78)
}

private struct HistoryThumbnail: View {
    @Environment(AppModel.self) private var app
    @Environment(\.undoManager) private var undoManager
    let item: HistoryItem
    let height: CGFloat
    let isSelected: Bool

    var body: some View {
        let url = app.url(for: item)
        FileImage(url: app.posterURL(for: item), maxPixelSize: 320)
            .aspectRatio(contentMode: .fill)
            .frame(width: thumbnailWidth(for: item.size, height: height), height: height)
            .background {
                if item.request.transparentBackground { Checkerboard(squareSize: 6) }
            }
            .overlay(alignment: .bottomLeading) {
                if item.kind == .video {
                    Image(systemName: "play.fill")
                        .font(.caption2)
                        .padding(4)
                        .background(.black.opacity(0.5), in: Circle())
                        .foregroundStyle(.white)
                        .padding(5)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(isSelected ? Color.active : Color.primary.opacity(0.1), lineWidth: isSelected ? 3 : 1)
            }
            .contentShape(Rectangle())
            .onTapGesture { app.select(item, undoManager: undoManager) }
            // The file, for Finder, other apps and the reference image; and the item itself, which
            // only this strip reads, to move it.
            .onDrag {
                let provider = NSItemProvider(contentsOf: url) ?? NSItemProvider()
                provider.registerDataRepresentation(for: .historyItem, visibility: .all) { completion in
                    completion(Data(item.id.uuidString.utf8), nil)
                    return nil
                }
                return provider
            }
            .onDrop(of: [.historyItem], delegate: HistoryMoveDelegate(target: item.id, app: app))
            .contextMenu { HistoryItemMenu(item: item) }
            .help(item.prompt)
    }
}

/// Moves the dragged item into place as it passes over the others, so the strip shows where it
/// will land; the drop only ends the drag. Which item it is comes with the drag itself: SwiftUI
/// does not run `onDrag` again for a view dragged before, so nothing set there can be relied on.
/// The item's id travels as `UTType.historyItem`; failing that, its file's name (unique in the
/// history) names it.
private struct HistoryMoveDelegate: DropDelegate {
    let target: HistoryItem.ID
    let app: AppModel

    func validateDrop(info: DropInfo) -> Bool {
        info.hasItemsConforming(to: [.historyItem]) || namedItem(info) != nil
    }

    func dropEntered(info: DropInfo) {
        let (target, app) = (target, app)
        if let provider = info.itemProviders(for: [.historyItem]).first {
            _ = provider.loadDataRepresentation(for: .historyItem) { data, _ in
                guard let data, let dragged = UUID(uuidString: String(decoding: data, as: UTF8.self)) else { return }
                Task { @MainActor in move(dragged, to: target, app: app) }
            }
        } else if let dragged = namedItem(info) {
            move(dragged, to: target, app: app)
        }
    }

    func dropUpdated(info: DropInfo) -> DropProposal? {
        DropProposal(operation: .move)
    }

    func performDrop(info: DropInfo) -> Bool {
        true
    }

    /// The history item whose file the drag carries, by name.
    private func namedItem(_ info: DropInfo) -> HistoryItem.ID? {
        let names = Set(info.itemProviders(for: [.item]).compactMap(\.suggestedName))
        guard !names.isEmpty else { return nil }
        return app.history.items.first { item in
            names.contains(item.fileName) || names.contains((item.fileName as NSString).deletingPathExtension)
        }?.id
    }

    @MainActor
    private func move(_ dragged: HistoryItem.ID, to target: HistoryItem.ID, app: AppModel) {
        guard dragged != target else { return }
        withAnimation(.snappy) { app.moveHistoryItem(dragged, to: target) }
    }
}

/// A button, as the "+" is: a tap gesture would miss the click that brings the window forward.
private struct JobThumbnail: View {
    @Environment(AppModel.self) private var app
    @Environment(\.undoManager) private var undoManager
    let job: GenerationJob
    let height: CGFloat
    let isSelected: Bool

    var body: some View {
        Button { app.select(job, undoManager: undoManager) } label: {
            RoundedRectangle(cornerRadius: 6)
                .fill(.quaternary.opacity(0.6))
                .overlay {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 6).strokeBorder(Color.active, lineWidth: 3)
                    } else {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(.tertiary, style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    }
                }
                .overlay {
                    VStack(spacing: 6) {
                        if job.phase == .queued {
                            Image(systemName: "clock")
                                .foregroundStyle(.secondary)
                            Text("Queued").font(.caption2).foregroundStyle(.secondary)
                        } else if let fraction = job.fraction {
                            ProgressView(value: fraction)
                                .progressViewStyle(.circular)
                                .controlSize(.small)
                            Text("\(job.step)/\(job.totalSteps)")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        } else {
                            ProgressView().controlSize(.small)
                        }
                    }
                }
                .frame(width: thumbnailWidth(for: job.request.size, height: height), height: height)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(job.phase == .queued ? "Remove from Queue" : "Stop") { app.cancel(job) }
        }
        .help(job.request.prompt)
        .accessibilityLabel(job.statusLabel)
    }
}
