import SwiftUI

/// Shown instead of an empty history when the selected model is not on disk yet.
struct ModelDownloadCard: View {
    @Environment(AppModel.self) private var app
    let model: ModelDescriptor

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "square.and.arrow.down.on.square")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)
            VStack(spacing: 6) {
                Text(model.name)
                    .font(.title2.bold())
                Text(model.detail)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }
            if let progress = app.downloads.active[model.id] {
                VStack(spacing: 6) {
                    ProgressView(value: progress.fraction ?? 0)
                        .frame(maxWidth: 360)
                    Text("\(Format.bytes(progress.bytes)) of \(Format.bytes(progress.total))"
                        + (progress.secondsRemaining.map { " · \(Format.remaining($0))" } ?? ""))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            } else {
                Button {
                    app.download(model)
                } label: {
                    Label(model.sizeBytes.map { "Download · \(Format.bytes($0))" } ?? "Download", systemImage: "arrow.down.circle")
                        .padding(.horizontal, 8)
                }
                .buttonStyle(.primaryAction)
                .disabled(model.repo == nil)
                Text("Downloaded once from Hugging Face\(model.license.map { " · \($0) license" } ?? ""). You can pick another model in the menu on the left.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(minWidth: 320, maxWidth: 520)
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Toolbar indicator with the engine's state; the popover has details and controls.
struct BackendStatusButton: View {
    @Environment(AppModel.self) private var app
    @State private var showsDetails = false

    var body: some View {
        Button {
            showsDetails.toggle()
        } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(color)
                    .frame(width: 8, height: 8)
                Text(title)
                    .font(.callout)
            }
            .padding(.horizontal, 4)
        }
        .help("Image engine status")
        .popover(isPresented: $showsDetails, arrowEdge: .bottom) {
            BackendDetails()
        }
    }

    private var color: Color {
        switch app.backend.status {
        case .ready: .green
        case .starting, .checking: .yellow
        case .stopped: .gray
        case .failed: .red
        }
    }

    private var title: String {
        switch app.backend.status {
        case .checking: "Starting…"
        case .starting: "Starting the engine…"
        case .ready: app.backend.loadedModelPath == nil ? "Ready" : "Ready · model in memory"
        case .stopped: "Engine stopped"
        case .failed: "Engine error"
        }
    }
}

struct BackendDetails: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let backend = app.backend
        VStack(alignment: .leading, spacing: 12) {
            Text("Image Engine").font(.headline)
            if let info = backend.info {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    GridRow { Text("Mac").foregroundStyle(.secondary); Text("\(info.device) · \(Format.memory(info.memory))") }
                    GridRow { Text("Engine").foregroundStyle(.secondary); Text(info.engine) }
                    GridRow { Text("MLX").foregroundStyle(.secondary); Text(info.mlx) }
                    GridRow {
                        Text("In memory").foregroundStyle(.secondary)
                        Text(loadedModelName ?? "no model")
                    }
                }
                .font(.callout)
            } else if case .failed(let message) = backend.status {
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                    .frame(maxWidth: 320, alignment: .leading)
            } else {
                Text("The engine is not running.").foregroundStyle(.secondary)
            }

            Divider()

            HStack {
                Button("Restart") { backend.restartWorker() }
                    .disabled(app.isBusy)
                Button("Free Memory") { app.freeMemory() }
                    .disabled(backend.loadedModelPath == nil || app.isBusy)
                Spacer()
                Button("Log…") { openWindow(id: WindowID.log) }
            }
        }
        .padding(16)
        .frame(width: 360)
    }

    private var loadedModelName: String? {
        guard let path = app.backend.loadedModelPath else { return nil }
        return app.models.first { app.installed[$0.id]?.path == path }?.name ?? URL(fileURLWithPath: path).lastPathComponent
    }
}
