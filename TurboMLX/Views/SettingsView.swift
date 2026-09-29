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
                    LabeledContent("Engine", value: info.engine)
                    LabeledContent("MLX", value: info.mlx)
                } else {
                    LabeledContent("Status", value: statusText)
                }
                LabeledContent("Executable") {
                    Text(BackendController.engineURL?.path(percentEncoded: false) ?? "Missing")
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .foregroundStyle(.secondary)
                }
            } footer: {
                Text("MLX Swift, part of the app: every model runs on it, and nothing else is installed.")
                    .foregroundStyle(.secondary)
            }

            Section {
                HStack {
                    Button("Restart Engine") { backend.restartWorker() }
                        .disabled(app.isBusy)
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
        case .ready: "Ready"
        case .stopped: "Stopped"
        case .failed(let message): message
        }
    }
}

private struct ModelsSettings: View {
    @Environment(AppModel.self) private var app
    @State private var modelToTrash: ModelDescriptor?

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
                                // Also what resumes a download stopped halfway.
                                Button("Download") { app.download(model) }
                            }
                            if app.downloadFiles(of: model) != nil {
                                Button {
                                    modelToTrash = model
                                } label: {
                                    Image(systemName: "trash")
                                }
                                .buttonStyle(.borderless)
                                .disabled(!app.canTrash(model))
                                .help("Move the downloaded files to the Trash")
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
            }
            ModelsFolderSection()
            HuggingFaceTokenSection()
        }
        .formStyle(.grouped)
        .confirmsTrashing($modelToTrash)
    }

    private func subtitle(for model: ModelDescriptor) -> String {
        let state = if app.isInstalled(model) {
            "Downloaded"
        } else if let files = app.downloadFiles(of: model), !app.downloads.isDownloading(model) {
            "Partly downloaded (\(Format.bytes(files.bytes)))"
        } else {
            "Not downloaded"
        }
        let size = model.sizeBytes.map { " · \(Format.bytes($0))" } ?? ""
        return "\(state)\(size) · \(model.id)"
    }
}

/// Where the downloaded models are kept, and another folder to keep them in.
private struct ModelsFolderSection: View {
    @Environment(AppModel.self) private var app
    @State private var problem: String?

    var body: some View {
        Section {
            LabeledContent("Folder") {
                Text(app.modelsFolder.path(percentEncoded: false))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            HStack {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([app.modelsFolder]) }
                    .disabled(!FileManager.default.fileExists(atPath: app.modelsFolder.path))
                Spacer()
                if ModelFolder.custom != nil {
                    Button("Use Default") { use(nil) }
                }
                Button("Change…") { choose() }
            }
            // A download writes into the folder and the engine reads from it: both must be idle.
            .disabled(app.isBusy || !app.downloads.active.isEmpty)
        } header: {
            Text("Models Folder")
        } footer: {
            Text(note)
                .foregroundStyle(.secondary)
        }
        .alert("The folder cannot be used", isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })) {
        } message: {
            Text(problem ?? "")
        }
    }

    private var note: String {
        if let missing = ModelFolder.unavailable {
            return "\(missing) could not be opened (a disk not connected?): until it can, models go to the default folder."
        }
        if ModelFolder.custom != nil {
            return "The folder you chose: models download there and are looked for there. Those in the default folder stay where they are."
        }
        #if APPSTORE
        return "A folder in the app’s data. To use the models mflux or the huggingface CLI downloaded, choose their folder, ~/.cache/huggingface (⌘⇧. shows hidden folders in the panel)."
        #else
        return "The shared Hugging Face cache, the same one mflux and other tools use. Another folder, on an external disk for instance, can take its place."
        #endif
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.showsHiddenFiles = true
        panel.prompt = "Use This Folder"
        panel.message = "Choose where to keep the models. A folder with models from mflux or the huggingface CLI (its hub, or the folder around it) is used as it is."
        panel.directoryURL = app.modelsFolder.deletingLastPathComponent()
        guard panel.runModal() == .OK, let url = panel.url else { return }
        use(url)
    }

    private func use(_ folder: URL?) {
        do {
            try app.useModelsFolder(folder)
        } catch {
            problem = error.localizedDescription
        }
    }
}

/// The Hugging Face token, for gated or private repositories, kept in the keychain.
private struct HuggingFaceTokenSection: View {
    @State private var token = ""
    @State private var isSaved = HuggingFaceToken.saved != nil
    @State private var problem: String?

    var body: some View {
        Section {
            SecureField("Access token", text: $token, prompt: Text(isSaved ? "Saved in the keychain" : "hf_…"))
            HStack {
                Spacer()
                if isSaved {
                    Button("Remove") {
                        HuggingFaceToken.remove()
                        isSaved = false
                    }
                }
                Button("Save") { save() }
                    .disabled(token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        } header: {
            Text("Hugging Face Token")
        } footer: {
            Text(note)
                .foregroundStyle(.secondary)
        }
        .alert("The token was not saved", isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })) {
        } message: {
            Text(problem ?? "")
        }
    }

    private var note: String {
        #if APPSTORE
        "Only for gated or private repositories added with Add Model…: the models of the list need none. It is kept in your keychain."
        #else
        "Only for gated or private repositories added with Add Model…: the models of the list need none. It is kept in your keychain; without one, the token the huggingface CLI saved is used."
        #endif
    }

    private func save() {
        do {
            try HuggingFaceToken.save(token.trimmingCharacters(in: .whitespacesAndNewlines))
            token = ""
            isSaved = true
        } catch {
            problem = error.localizedDescription
        }
    }
}

private struct HistorySettings: View {
    @Environment(AppModel.self) private var app
    @State private var confirmsClear = false

    var body: some View {
        Form {
            Section {
                LabeledContent("Images and clips", value: "\(app.history.items.count)")
                LabeledContent("Folder") {
                    Button("Show in Finder") { NSWorkspace.shared.open(HistoryStore.directory) }
                }
            } footer: {
                Text("Each image is a PNG with its generation metadata embedded, each clip an MP4 next to a PNG of its first frame; the history index is history.json in the same folder.")
                    .foregroundStyle(.secondary)
            }
            Section {
                Button("Clear History…", role: .destructive) { confirmsClear = true }
                    .disabled(app.history.items.isEmpty)
            }
        }
        .formStyle(.grouped)
        .confirmationDialog("Clear the history?", isPresented: $confirmsClear) {
            Button("Move \(app.history.items.count) Items to the Trash", role: .destructive) { app.clearHistory() }
        } message: {
            Text("The images and clips go to the Trash, so you can still recover them from Finder.")
        }
    }
}

private struct AboutSettings: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Form {
            Section {
                LabeledContent("Turbo MLX", value: Self.version)
                Text("Generates images, and videos with sound, on your Mac’s GPU with MLX. Models download from Hugging Face and stay on your computer: prompts, images and videos never leave your Mac.")
                    .foregroundStyle(.secondary)
                LabeledContent("Source code") {
                    Link("MIT License · GitHub", destination: URL(string: "https://github.com/rinste/turbo-mlx")!)
                }
            }
            #if !APPSTORE
            updatesSection
            #endif
            Section("Components") {
                LabeledContent("MLX Swift", value: "MIT · Apple")
                LabeledContent("swift-transformers", value: "Apache 2.0 · Hugging Face")
                #if !APPSTORE
                LabeledContent("Sparkle", value: "MIT · the updates")
                #endif
                LabeledContent("mflux, ltx-2-mlx", value: "MIT · the references the engine follows")
                Button("Acknowledgements…") { openWindow(id: WindowID.acknowledgements) }
            }
            Section {
                ForEach(app.models.filter(\.isBuiltIn)) { model in
                    LabeledContent(model.name) { license(model.license ?? "—", model.webURL) }
                }
                if let companion = ModelFamily.ltx2.companion {
                    LabeledContent("\(companion.name), LTX-2.3’s text encoder") {
                        license(companion.license, URL(string: "https://ai.google.dev/gemma/terms"))
                    }
                }
            } header: {
                Text("Models")
            } footer: {
                Text("Each model comes from Hugging Face under its own license, whose page the link opens.")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    #if !APPSTORE
    /// Sparkle's settings, in the GitHub build only.
    private var updatesSection: some View {
        let updater = app.updater
        return Section {
            Toggle("Check for updates automatically", isOn: Binding(
                get: { updater.checksAutomatically }, set: { updater.setChecksAutomatically($0) }))
            Toggle("Download and install them without asking", isOn: Binding(
                get: { updater.installsAutomatically }, set: { updater.setInstallsAutomatically($0) }))
                .disabled(!updater.checksAutomatically)
            HStack {
                Spacer()
                Button("Check Now") { updater.checkForUpdates() }
                    .disabled(!updater.canCheckForUpdates)
            }
        } footer: {
            Text("Once a day the app asks GitHub for the latest release. A new version is installed only if it carries the app’s signature, and the app restarts with it; images, settings and models stay.")
                .foregroundStyle(.secondary)
        }
        .onAppear { updater.refresh() }
    }
    #endif

    /// "1.1 (2)": the version people see, and the build number updates are compared by.
    private static var version: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        return (info?["CFBundleVersion"] as? String).map { "\(short) (\($0))" } ?? short
    }

    @ViewBuilder
    private func license(_ name: String, _ url: URL?) -> some View {
        if let url { Link(name, destination: url) } else { Text(name) }
    }
}

/// The licenses of the code in the app, in a window of its own (Settings → About):
/// Resources/Acknowledgements.txt, written by scripts/make-acknowledgements.swift.
struct AcknowledgementsView: View {
    private let text = Bundle.main.url(forResource: "Acknowledgements", withExtension: "txt")
        .flatMap { try? String(contentsOf: $0, encoding: .utf8) }
        ?? "The acknowledgements are missing from this copy of Turbo MLX."

    var body: some View {
        ScrollView {
            Text(text)
                .font(.system(size: 11, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
        }
    }
}
