import SwiftUI

/// Left column: model, prompt and generation settings.
struct ControlPanel: View {
    @Environment(AppModel.self) private var app
    @FocusState private var promptFocused: Bool

    var body: some View {
        @Bindable var app = app
        Form {
            ModelSection()
            if let model = app.selectedModel {
                PromptSection(prompt: $app.settings.prompt, family: model.family, isFocused: $promptFocused)
                FormatSection(settings: $app.settings)
                ParametersSection(settings: $app.settings, model: model)
                MemorySection(settings: $app.settings, family: model.family)
            }
        }
        .formStyle(.grouped)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            GenerateBar()
        }
        .onAppear { promptFocused = true }
    }
}

// MARK: - Model

private struct ModelSection: View {
    @Environment(AppModel.self) private var app
    @State private var showsAddModel = false

    var body: some View {
        @Bindable var app = app
        Section {
            Picker("Model", selection: $app.selectedModelID) {
                ForEach(ModelFamily.allCases, id: \.self) { family in
                    let members = app.models.filter { $0.family == family }
                    if !members.isEmpty {
                        Section(family.displayName) {
                            ForEach(members) { model in
                                Label(model.name, systemImage: app.isInstalled(model) ? "checkmark.circle.fill" : "arrow.down.circle")
                                    .tag(model.id)
                            }
                        }
                    }
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)

            if let model = app.selectedModel {
                VStack(alignment: .leading, spacing: 8) {
                    Text(model.detail)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    ModelStatusRow(model: model)
                    MemoryWarning(model: model)
                }
                .padding(.vertical, 2)
            }
        } header: {
            HStack {
                Text("Model")
                Spacer()
                ModelMenu(showsAddModel: $showsAddModel)
            }
        }
        .sheet(isPresented: $showsAddModel) {
            AddModelSheet()
        }
    }
}

private struct ModelStatusRow: View {
    @Environment(AppModel.self) private var app
    let model: ModelDescriptor

    var body: some View {
        if let progress = app.downloads.active[model.id] {
            DownloadProgressView(model: model, progress: progress)
        } else if app.isInstalled(model) {
            HStack(spacing: 8) {
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                if app.isLoaded(model) {
                    Label("In memory", systemImage: "memorychip")
                        .foregroundStyle(.secondary)
                        .help("The model is already loaded: the next image starts right away.")
                }
                Spacer()
                Text(footnote).foregroundStyle(.tertiary)
            }
            .font(.callout)
        } else {
            VStack(alignment: .leading, spacing: 6) {
                Button {
                    app.download(model)
                } label: {
                    Label(
                        model.sizeBytes.map { "Download · \(Format.bytes($0))" } ?? "Download",
                        systemImage: "arrow.down.circle"
                    )
                    .frame(maxWidth: .infinity)
                }
                .controlSize(.large)
                .disabled(model.repo == nil || !app.backend.isInstalled)
                if let failure = app.downloads.failures[model.id] {
                    Text(failure)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(3)
                        .textSelection(.enabled)
                } else if model.repo == nil {
                    Text("This folder doesn’t contain a complete model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if let license = model.license {
                    Text("\(license) license · downloaded once from Hugging Face")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    private var footnote: String {
        [model.sizeBytes.map(Format.bytes), model.license].compactMap(\.self).joined(separator: " · ")
    }
}

/// Warns before a download that this Mac has less memory than the model wants.
private struct MemoryWarning: View {
    let model: ModelDescriptor

    var body: some View {
        let installedGB = Int(ProcessInfo.processInfo.physicalMemory >> 30)
        if let recommended = model.recommendedMemoryGB, installedGB < recommended {
            Label(
                "Made for Macs with at least \(recommended) GB of memory; this one has \(installedGB) GB. It may work, but slowly.",
                systemImage: "exclamationmark.triangle"
            )
            .font(.caption)
            .foregroundStyle(.orange)
            .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct DownloadProgressView: View {
    @Environment(AppModel.self) private var app
    let model: ModelDescriptor
    let progress: DownloadCenter.Progress

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                if let fraction = progress.fraction {
                    ProgressView(value: fraction)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
                Button {
                    app.cancelDownload(model)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .help("Cancel the download (you can resume it later)")
            }
            HStack {
                Text("\(Format.bytes(progress.bytes)) of \(Format.bytes(progress.total))")
                Spacer()
                if progress.bytesPerSecond > 0 {
                    Text("\(Format.bytes(Int64(progress.bytesPerSecond)))/s")
                }
                if let remaining = progress.secondsRemaining {
                    Text("· \(Format.remaining(remaining))")
                }
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            if app.generateAfterDownload == model.id {
                Label("The image will be generated as soon as the download finishes.", systemImage: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct ModelMenu: View {
    @Environment(AppModel.self) private var app
    @Binding var showsAddModel: Bool

    var body: some View {
        Menu {
            if let model = app.selectedModel {
                if let url = model.webURL {
                    Link("Open on Hugging Face", destination: url)
                }
                if let location = app.installed[model.id] {
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([location])
                    }
                }
                if !model.isBuiltIn {
                    Button("Remove from List", role: .destructive) {
                        app.removeCustomModel(model)
                    }
                }
                Divider()
            }
            Button("Add Model…") { showsAddModel = true }
            Button("Refresh Model Status") { app.refreshInstalled() }
        } label: {
            Image(systemName: "ellipsis.circle")
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .textCase(nil)
    }
}

// MARK: - Prompt

private struct PromptSection: View {
    @Binding var prompt: String
    let family: ModelFamily
    var isFocused: FocusState<Bool>.Binding

    var body: some View {
        Section {
            TextEditor(text: $prompt)
                .font(.body)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 110, idealHeight: 140, maxHeight: 280)
                .focused(isFocused)
                .overlay(alignment: .topLeading) {
                    if prompt.isEmpty {
                        Text("Describe the image: subject, style, light, colors. Put any text to render in quotes.")
                            .foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
        } header: {
            HStack {
                Text("Prompt")
                Spacer()
                Menu("Examples") {
                    ForEach(PromptExamples.for(family), id: \.title) { example in
                        Button(example.title) { prompt = example.prompt }
                    }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .textCase(nil)
            }
        } footer: {
            Text(family.promptTip)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Format

private struct FormatSection: View {
    @Binding var settings: GenerationSettings

    var body: some View {
        Section("Format") {
            HStack(spacing: 2) {
                ForEach(AspectRatio.allCases) { aspect in
                    let isSelected = !settings.usesCustomSize && settings.aspect == aspect
                    Button {
                        settings.aspect = aspect
                        settings.usesCustomSize = false
                    } label: {
                        VStack(spacing: 4) {
                            AspectGlyph(ratio: aspect.ratio, isSelected: isSelected)
                            Text(aspect.rawValue)
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(isSelected ? .primary : .secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 5)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(isSelected ? Color.accentColor.opacity(0.12) : .clear)
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("\(aspect.rawValue) · \(aspect.usage)")
                }
            }

            Picker("Resolution", selection: $settings.resolution) {
                ForEach(GenerationSettings.resolutions, id: \.self) { resolution in
                    Text(verbatim: "\(resolution)").tag(resolution)
                }
            }
            .pickerStyle(.segmented)
            .disabled(settings.usesCustomSize)

            Toggle("Custom size", isOn: $settings.usesCustomSize)

            if settings.usesCustomSize {
                HStack {
                    TextField("Width", value: $settings.customWidth, format: .number.grouping(.never))
                    Text(verbatim: "×").foregroundStyle(.secondary)
                    TextField("Height", value: $settings.customHeight, format: .number.grouping(.never))
                }
                .multilineTextAlignment(.center)
                .labelsHidden()
            }

            let size = settings.size
            LabeledContent("Image") {
                Text(verbatim: "\(size.width) × \(size.height) px · \(size.megapixels.formatted(.number.precision(.fractionLength(1)))) MP")
                    .monospacedDigit()
            }
            if size.megapixels > 2.5 {
                Label("High resolutions take much more time and memory.", systemImage: "tortoise")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }
}

// MARK: - Parameters

private struct ParametersSection: View {
    @Binding var settings: GenerationSettings
    let model: ModelDescriptor

    var body: some View {
        let range = model.stepRange
        Section("Parameters") {
            LabeledContent {
                HStack(spacing: 6) {
                    // Steps trade speed for refinement: hare for fewer, tortoise for more.
                    Image(systemName: "hare")
                        .foregroundStyle(.secondary)
                        .help("Fewer steps: faster")
                        .accessibilityLabel("Faster")
                    Slider(
                        value: Binding(
                            get: { Double(min(max(settings.steps, range.lowerBound), range.upperBound)) },
                            set: { settings.steps = Int($0.rounded()) }
                        ),
                        in: Double(range.lowerBound)...Double(range.upperBound),
                        step: 1
                    )
                    Image(systemName: "tortoise")
                        .foregroundStyle(.secondary)
                        .help("More steps: slower, sometimes more refined")
                        .accessibilityLabel("Slower")
                    Text(verbatim: "\(settings.steps)")
                        .monospacedDigit()
                        .frame(width: 30, alignment: .trailing)
                }
            } label: {
                Text("Steps")
                    .help("Recommended for this model: \(model.defaultSteps). More steps take longer and don’t always improve the result.")
            }

            if model.supportsGuidance {
                LabeledContent {
                    HStack(spacing: 6) {
                        // Invisible stand-ins for the steps slider's icons keep both sliders aligned.
                        Image(systemName: "hare").hidden()
                        Slider(value: $settings.guidance, in: GenerationSettings.guidanceRange, step: 0.5)
                        Image(systemName: "tortoise").hidden()
                        Text(Format.guidance(settings.guidance))
                            .monospacedDigit()
                            .frame(width: 30, alignment: .trailing)
                    }
                } label: {
                    Text("Guidance")
                        .help(model.family == .qwenImage
                              ? "How strictly to follow the prompt. 4 is the recommended value."
                              : "How strictly to follow the prompt. 1 = off; higher values double the time of each step.")
                }
            }

            LabeledContent("Seed") {
                HStack(spacing: 8) {
                    TextField("Seed", value: $settings.seed, format: .number.grouping(.never))
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .disabled(settings.randomSeed)
                        .frame(maxWidth: 120)
                    Toggle("Random", isOn: $settings.randomSeed)
                        .toggleStyle(.checkbox)
                }
            }

            LabeledContent {
                HStack(spacing: 8) {
                    Text(verbatim: "\(settings.batchCount)")
                        .monospacedDigit()
                    Stepper("Outputs", value: $settings.batchCount, in: 1...GenerationSettings.maxBatch)
                        .labelsHidden()
                }
            } label: {
                Text("Outputs")
                    .help("How many images one click generates, each with its own seed: consecutive from a fixed seed, or random.")
            }

            if model.family.producesAlpha {
                Picker("Background", selection: $settings.transparentBackground) {
                    Text("Transparent").tag(true)
                    Text("White").tag(false)
                }
                .pickerStyle(.segmented)
            }
        }
    }
}

// MARK: - Memory

private struct MemorySection: View {
    @Environment(AppModel.self) private var app
    @Binding var settings: GenerationSettings
    let family: ModelFamily

    private var saveMemoryDescription: String {
        let base = switch family {
        case .ming:
            "Frees the text encoder (about 12 GB) once the prompt is read and decodes the image in tiles: about 15 GB instead of 35 GB at 1024 px. A new prompt reloads the model."
        case .qwenImage:
            "Frees the text encoder (about 14 GB) once the prompt is read and decodes the image in tiles. A new prompt reloads the model."
        case .zImageTurbo, .flux2Klein:
            "Keeps less in memory and, where it doesn’t affect the image, decodes it in tiles."
        }
        return base + " On by default below 64 GB of memory."
    }

    var body: some View {
        Section("Memory") {
            Toggle(isOn: $settings.lowMemory) {
                Text("Save memory")
                Text(saveMemoryDescription)
            }
            if app.backend.loadedModelPath != nil {
                LabeledContent {
                    Button("Free Memory") { app.backend.unloadModel() }
                        .disabled(app.activeJob != nil)
                } label: {
                    Text("Model loaded")
                    Text("It stays in memory so the next image starts right away.")
                }
            }
        }
    }
}

// MARK: - Generate

/// The main action adapts to what is missing: engine, model, then generation.
private struct GenerateBar: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 10) {
            if let job = app.activeJob {
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(job.statusLabel)
                            Spacer()
                            if let remaining = job.estimatedSecondsRemaining {
                                Text(Format.remaining(remaining)).foregroundStyle(.secondary)
                            }
                        }
                        .font(.caption.monospacedDigit())
                        if let fraction = job.fraction {
                            ProgressView(value: fraction)
                        } else {
                            ProgressView().progressViewStyle(.linear)
                        }
                    }
                    Button {
                        app.cancel(job)
                    } label: {
                        Image(systemName: job.isCancelling ? "xmark.octagon.fill" : "stop.fill")
                    }
                    .help(job.isCancelling ? "Force stop (the model will be reloaded)" : "Stop (⌘.)")
                }
            }

            primaryButton
                .buttonStyle(.borderedProminent)
                .controlSize(.extraLarge)

            if let caption {
                Text(caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(14)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    @ViewBuilder
    private var primaryButton: some View {
        let model = app.selectedModel
        switch app.blocker {
        case .backendNotInstalled:
            wideButton("Install Image Engine", systemImage: "shippingbox") { app.backend.install() }
        case .backendInstalling:
            wideButton("Installing Image Engine…", systemImage: "shippingbox") {}
                .disabled(true)
        case .modelNotDownloaded:
            let size = model?.sizeBytes.map { " · \(Format.bytes($0))" } ?? ""
            if app.settings.trimmedPrompt.isEmpty {
                wideButton("Download Model\(size)", systemImage: "arrow.down.circle") {
                    if let model { app.download(model) }
                }
                .disabled(model?.repo == nil)
            } else {
                wideButton(app.isBusy ? "Download and Queue\(size)" : "Download and Generate\(size)",
                           systemImage: "arrow.down.circle") { app.downloadAndGenerate() }
                    .disabled(model?.repo == nil)
            }
        case .modelDownloading:
            let percent = model.flatMap { app.downloads.active[$0.id]?.fraction }.map { " · \(Int($0 * 100))%" } ?? ""
            wideButton("Downloading\(percent)", systemImage: "arrow.down.circle") {}
                .disabled(true)
        case .noModel, .emptyPrompt:
            wideButton(generateTitle, systemImage: generateSymbol, shortcut: "⌘↩") {}
                .disabled(true)
        case nil:
            wideButton(generateTitle, systemImage: generateSymbol, shortcut: "⌘↩") { app.generate() }
                .help(app.isBusy
                      ? "Queue these settings: they start when the images ahead are done (⌘↩)"
                      : "Generate (⌘↩)")
        }
    }

    /// Prompt, model and settings are captured now, so the user can keep editing while it waits.
    private var generateTitle: String {
        let count = app.settings.batchCount
        if app.isBusy { return count > 1 ? "Add \(count) to Queue" : "Add to Queue" }
        return count > 1 ? "Generate \(count) Images" : "Generate"
    }

    private var generateSymbol: String {
        app.isBusy ? "text.badge.plus" : "sparkles"
    }

    private var caption: String? {
        switch app.blocker {
        case .backendNotInstalled: "One time only: about 1 GB, a few minutes."
        case .modelNotDownloaded: "Models download once and stay on this Mac."
        case .some(let blocker): blocker.hint
        case nil:
            switch app.pendingJobs.count {
            case 0: nil
            case 1: "Starts when the current image is done"
            case let ahead: "Starts after the \(ahead) images ahead of it"
            }
        }
    }

    /// `shortcut` follows the title, lighter, e.g. "Generate (⌘↩)".
    private func wideButton(
        _ title: String,
        systemImage: String,
        shortcut: String? = nil,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label {
                if let shortcut {
                    Text("\(title) \(Text(verbatim: "(\(shortcut))").fontWeight(.regular).foregroundStyle(.secondary))")
                        .accessibilityLabel(title)
                } else {
                    Text(title)
                }
            } icon: {
                Image(systemName: systemImage)
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
        }
    }
}
