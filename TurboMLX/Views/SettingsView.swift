import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            Tab("Engine", systemImage: "cpu") { EngineSettings() }
            Tab("Models", systemImage: "square.stack.3d.up") { ModelsSettings() }
            Tab("History", systemImage: "clock.arrow.circlepath") { HistorySettings() }
            Tab("About", systemImage: "info.circle") { AboutSettings() }
        }
        .frame(width: 560, height: 440)
    }
}

private struct EngineSettings: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let backend = app.backend
        Form {
            Section {
                if let info = backend.info {
                    LabeledContent("Mac", value: "\(info.device) · \(Format.memory(info.memory))")
                    LabeledContent("mflux", value: info.mflux)
                    LabeledContent("MLX", value: info.mlx)
                    LabeledContent("Python", value: info.python)
                } else {
                    LabeledContent("Status", value: statusText)
                }
            }

            Section {
                LabeledContent("Python environment") {
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([BackendController.venvDirectory])
                    }
                    .disabled(!backend.isInstalled)
                }
                LabeledContent("mflux version") {
                    Text(verbatim: "commit " + BackendController.mfluxCommit.prefix(7))
                        .monospaced()
                        .foregroundStyle(.secondary)
                }
            } footer: {
                Text("Repair rebuilds the app’s Python environment from scratch, reusing the packages already downloaded. Use it if the engine no longer starts.")
                    .foregroundStyle(.secondary)
            }

            Section {
                HStack {
                    Button(backend.isInstalled ? "Repair Engine" : "Install Engine") {
                        backend.install(clean: backend.isInstalled)
                    }
                    .disabled(app.isBusy || backend.status == .installing)
                    Button("Restart Engine") { backend.restartWorker() }
                        .disabled(!backend.isInstalled || app.isBusy)
                    Spacer()
                    Button("Log…") { openWindow(id: WindowID.log) }
                }
            }
        }
        .formStyle(.grouped)
    }

    private var statusText: String {
        switch app.backend.status {
        case .checking, .starting: "Starting…"
        case .notInstalled: "Not installed"
        case .installing: "Installing…"
        case .ready: "Ready"
        case .stopped: "Stopped"
        case .failed(let message): message
        }
    }
}

private struct ModelsSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        Form {
            Section {
                ForEach(app.models) { model in
                    LabeledContent {
                        HStack(spacing: 8) {
                            if let progress = app.downloads.active[model.id] {
                                ProgressView(value: progress.fraction ?? 0)
                                    .frame(width: 80)
                                Button("Cancel") { app.cancelDownload(model) }
                            } else if let location = app.installed[model.id] {
                                Button("Show") { NSWorkspace.shared.activateFileViewerSelecting([location]) }
                            } else if model.repo != nil {
                                Button("Download") { app.download(model) }
                            }
                            if !model.isBuiltIn {
                                Button {
                                    app.removeCustomModel(model)
                                } label: {
                                    Image(systemName: "minus.circle")
                                }
                                .buttonStyle(.borderless)
                                .help("Remove from the list (the files stay on disk)")
                            }
                        }
                    } label: {
                        Text(model.name)
                        Text(subtitle(for: model))
                    }
                }
            } footer: {
                Text("Hugging Face models are stored in the shared cache (\(app.locator.hubCache.path(percentEncoded: false))), the same one mflux and other tools use.")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private func subtitle(for model: ModelDescriptor) -> String {
        let state = app.isInstalled(model) ? "Downloaded" : "Not downloaded"
        let size = model.sizeBytes.map { " · \(Format.bytes($0))" } ?? ""
        return "\(state)\(size) · \(model.id)"
    }
}

private struct HistorySettings: View {
    @Environment(AppModel.self) private var app
    @State private var confirmsClear = false

    var body: some View {
        Form {
            Section {
                LabeledContent("Images", value: "\(app.history.items.count)")
                LabeledContent("Folder") {
                    Button("Show in Finder") { NSWorkspace.shared.open(HistoryStore.directory) }
                }
            } footer: {
                Text("Each image is a PNG with its generation metadata embedded; the history index is history.json in the same folder.")
                    .foregroundStyle(.secondary)
            }
            Section {
                Button("Clear History…", role: .destructive) { confirmsClear = true }
                    .disabled(app.history.items.isEmpty)
            }
        }
        .formStyle(.grouped)
        .confirmationDialog("Clear the history?", isPresented: $confirmsClear) {
            Button("Move \(app.history.items.count) Images to the Trash", role: .destructive) { app.clearHistory() }
        } message: {
            Text("The images go to the Trash, so you can still recover them from Finder.")
        }
    }
}

private struct AboutSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        Form {
            Section {
                LabeledContent("Turbo MLX", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")
                Text("Generates images on your Mac’s GPU with MLX. Models download from Hugging Face and stay on your computer: prompts and images never leave your Mac.")
                    .foregroundStyle(.secondary)
            }
            Section("Components") {
                LabeledContent("mflux", value: "MIT · mflux-community")
                LabeledContent("MLX", value: "MIT · Apple")
                LabeledContent("uv", value: "MIT · Astral")
            }
            Section("Models") {
                ForEach(app.models.filter(\.isBuiltIn)) { model in
                    LabeledContent(model.name, value: model.license ?? "—")
                }
            }
        }
        .formStyle(.grouped)
    }
}
