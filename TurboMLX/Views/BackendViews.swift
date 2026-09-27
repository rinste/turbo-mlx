import SwiftUI

/// Shown in place of the image until the mflux environment exists.
struct BackendSetupView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openWindow) private var openWindow
    @State private var showsDetails = false

    var body: some View {
        let backend = app.backend
        VStack(spacing: 18) {
            Image(systemName: "shippingbox")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(.secondary)
            VStack(spacing: 8) {
                Text(backend.isUpdating ? "Updating the Image Engine" : "Set Up the Image Engine")
                    .font(.title2.bold())
                Text("Turbo MLX generates images right on your Mac, on the GPU. The first time, it downloads its engine (Python, MLX and mflux, about 1 GB) into a folder of its own: there’s nothing else to install, and the rest of your system is left untouched.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }

            switch backend.status {
            case .installing:
                VStack(spacing: 8) {
                    ProgressView()
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 360)
                    Text(backend.installPhase ?? "Starting the installation…")
                        .font(.callout)
                    Text("This usually takes a few minutes, depending on your connection.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            case .failed(let message):
                VStack(spacing: 10) {
                    Text(message)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .textSelection(.enabled)
                    HStack {
                        Button("Try Again") { backend.install() }
                            .buttonStyle(.borderedProminent)
                        Button("Open Log") { openWindow(id: WindowID.log) }
                    }
                }
            default:
                Button {
                    backend.install()
                } label: {
                    Label("Install Image Engine", systemImage: "arrow.down.circle")
                        .padding(.horizontal, 8)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }

            if !backend.installOutput.isEmpty {
                // Not a DisclosureGroup: collapsed, it still sized itself to the whole log.
                VStack(alignment: .leading, spacing: 8) {
                    Button {
                        withAnimation(.snappy) { showsDetails.toggle() }
                    } label: {
                        Label("Details", systemImage: showsDetails ? "chevron.down" : "chevron.right")
                    }
                    .buttonStyle(.borderless)
                    if showsDetails {
                        InstallLog(lines: backend.installOutput)
                    }
                }
                .frame(maxWidth: 580, alignment: .leading)
            }
        }
        .frame(minWidth: 320, maxWidth: 580)
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

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
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
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

private struct InstallLog: View {
    let lines: [String]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(lines.suffix(400).enumerated()), id: \.offset) { index, line in
                        Text(verbatim: line)
                            .font(.caption.monospaced())
                            .foregroundStyle(line.hasPrefix("==>") ? .primary : .secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(10)
                .textSelection(.enabled)
            }
            .frame(height: 200)
            .clipped()
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
            .onChange(of: lines.count) {
                proxy.scrollTo(min(lines.count, 400) - 1, anchor: .bottom)
            }
        }
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
        case .installing: .blue
        case .notInstalled: .orange
        case .stopped: .gray
        case .failed: .red
        }
    }

    private var title: String {
        switch app.backend.status {
        case .checking: "Starting…"
        case .notInstalled: "Engine not installed"
        case .installing: "Installing…"
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
                    GridRow { Text("mflux").foregroundStyle(.secondary); Text(info.mflux) }
                    GridRow { Text("MLX").foregroundStyle(.secondary); Text(info.mlx) }
                    GridRow { Text("Python").foregroundStyle(.secondary); Text(info.python) }
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
                if backend.isInstalled {
                    Button("Restart") { backend.restartWorker() }
                        .disabled(app.isBusy)
                    Button("Free Memory") { app.freeMemory() }
                        .disabled(backend.loadedModelPath == nil || app.isBusy)
                }
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
