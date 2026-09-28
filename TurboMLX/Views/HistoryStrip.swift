import SwiftUI

/// Film strip of past images and clips in the order they were made, the newest on the right, then
/// the running and queued jobs and a "+" that sets up the next one. A click shows an item or a job
/// and puts its settings on the left.
struct HistoryStrip: View {
    @Environment(AppModel.self) private var app
    @State private var confirmsClear = false
    /// At the end, where the newest are, until the user scrolls away.
    @State private var position = ScrollPosition(edge: .trailing)

    private let thumbnailHeight: CGFloat = 92

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text("History").font(.headline)
                Text("\(app.history.items.count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Spacer()
                if app.pendingJobs.count > 1 {
                    Button("Cancel All") { app.cancelAll() }
                        .buttonStyle(.link)
                        .font(.caption)
                }
                Menu {
                    Button("Show Folder in Finder") {
                        NSWorkspace.shared.open(HistoryStore.directory)
                    }
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
                        // The history is stored newest first.
                        ForEach(app.history.items.reversed()) { item in
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
            .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
            .contextMenu { HistoryItemMenu(item: item) }
            .help(item.prompt)
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
