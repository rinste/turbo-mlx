import SwiftUI

/// Film strip of queued jobs and past images, newest on the left.
struct HistoryStrip: View {
    @Environment(AppModel.self) private var app
    @State private var confirmsClear = false

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
                        // Newest on the left here too: the last queued job first, the running one
                        // next to the images it will join.
                        ForEach(app.pendingJobs.reversed()) { job in
                            JobThumbnail(job: job, height: thumbnailHeight)
                        }
                        ForEach(app.history.items) { item in
                            HistoryThumbnail(item: item, height: thumbnailHeight, isSelected: app.isSelected(item))
                                .id(item.id)
                        }
                        if app.history.items.isEmpty && app.pendingJobs.isEmpty {
                            Text("Generated images will appear here.")
                                .font(.callout)
                                .foregroundStyle(.tertiary)
                                .frame(height: thumbnailHeight)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                }
                .scrollIndicators(.automatic)
                .onChange(of: app.displayedItem?.id) { _, id in
                    guard let id else { return }
                    withAnimation(.snappy) { proxy.scrollTo(id, anchor: .center) }
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

private func thumbnailWidth(for size: PixelSize, height: CGFloat) -> CGFloat {
    let ratio = CGFloat(size.width) / CGFloat(max(size.height, 1))
    return min(max(height * ratio, height * 0.56), height * 1.78)
}

private struct HistoryThumbnail: View {
    @Environment(AppModel.self) private var app
    let item: HistoryItem
    let height: CGFloat
    let isSelected: Bool

    var body: some View {
        let url = app.url(for: item)
        FileImage(url: url, maxPixelSize: 320)
            .aspectRatio(contentMode: .fill)
            .frame(width: thumbnailWidth(for: item.size, height: height), height: height)
            .background {
                if item.request.transparentBackground { Checkerboard(squareSize: 6) }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(isSelected ? Color.accentColor : Color.primary.opacity(0.1), lineWidth: isSelected ? 3 : 1)
            }
            .contentShape(Rectangle())
            .onTapGesture { app.select(item) }
            .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
            .contextMenu { HistoryItemMenu(item: item) }
            .help(item.prompt)
    }
}

private struct JobThumbnail: View {
    @Environment(AppModel.self) private var app
    let job: GenerationJob
    let height: CGFloat

    var body: some View {
        RoundedRectangle(cornerRadius: 6)
            .fill(.quaternary.opacity(0.6))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(.tertiary, style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
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
            .onTapGesture { app.viewer = .live }
            .contextMenu {
                Button(job.phase == .queued ? "Remove from Queue" : "Stop") { app.cancel(job) }
            }
            .help(job.request.prompt)
    }
}
