import SwiftUI

/// Adds a checkpoint that is not in the built-in catalog: found by searching Hugging Face for the
/// ones Turbo MLX runs, or named by hand (a repository or a local folder).
struct AddModelSheet: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    private enum Origin: Hashable {
        case search
        case huggingFace
        case folder
    }

    @State private var origin = Origin.search
    @State private var family = ModelFamily.ming
    /// FLUX.2 Klein registry key; empty means "infer from the name".
    @State private var kleinVariant = ""
    @State private var repo = ""
    @State private var folder: URL?
    @State private var name = ""

    @State private var query = ""
    @State private var search = SearchState.idle
    /// Bumped by Try Again, to run the same search once more.
    @State private var searchAttempt = 0
    @State private var selectedResultID: HubModel.ID?
    /// What Hugging Face says of the repository typed by hand.
    @State private var repoCheck = RepoCheck.none

    private enum SearchState {
        case idle
        case loading
        case loaded([HubModel])
        case failed(String)
    }

    private enum RepoCheck: Equatable {
        case none
        case checking
        case found(ModelCompatibility.Verdict)
        case notFound
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Add a Model")
                    .font(.title3.bold())
                Text(origin == .search
                     ? "Search Hugging Face for checkpoints in mflux format. The ones Turbo MLX runs come first, with their family already set."
                     : "A checkpoint in mflux format (saved with mflux-save, or pre-quantized): a Hugging Face repository or a local folder.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Picker("Source", selection: $origin) {
                Text("Search").tag(Origin.search)
                Text("Repository").tag(Origin.huggingFace)
                Text("Local Folder").tag(Origin.folder)
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            switch origin {
            case .search: searchPane
            case .huggingFace, .folder: manualForm
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Add", action: add)
                    .keyboardShortcut(.defaultAction)
                    .disabled(source == nil)
            }
        }
        .padding(20)
        .frame(width: 560)
    }

    // MARK: Search

    private var searchPane: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextField("Search", text: $query, prompt: Text("Search: klein, qwen, z-image, ming…"))
                .textFieldStyle(.roundedBorder)
            resultList
                .frame(height: 300)
            if let result = selectedResult {
                selectionForm(result)
            }
        }
        .task(id: "\(query)\u{0}\(searchAttempt)") { await runSearch() }
    }

    @ViewBuilder
    private var resultList: some View {
        switch search {
        case .idle, .loading:
            ProgressView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let message):
            ContentUnavailableView {
                Label("Can't Search Hugging Face", systemImage: "wifi.exclamationmark")
            } description: {
                Text(message)
            } actions: {
                Button("Try Again") { searchAttempt += 1 }
            }
        case .loaded(let results) where results.isEmpty:
            ContentUnavailableView.search(text: query)
        case .loaded(let results):
            // Buttons rather than a List's selection: each row presses (also through accessibility),
            // and the ones Turbo MLX can't run stay unselectable.
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(results) { result in
                        let verdict = ModelCompatibility.assess(result)
                        ResultRow(result: result, verdict: verdict, isSelected: result.id == selectedResultID) {
                            selectedResultID = result.id
                        }
                        Divider()
                    }
                }
            }
            .background(.background.secondary)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.separator))
        }
    }

    /// The repositories in order of use: the ones that run, those whose family the user must pick,
    /// then the rest, each by downloads as the hub sorted them.
    private static func sorted(_ results: [HubModel]) -> [HubModel] {
        func rank(_ model: HubModel) -> Int {
            switch ModelCompatibility.assess(model) {
            case .supported: 0
            case .unknownFamily: 1
            case .unsupported: 2
            }
        }
        return results.enumerated()
            .sorted { (rank($0.element), $0.offset) < (rank($1.element), $1.offset) }
            .map(\.element)
    }

    private func runSearch() async {
        // Typing restarts the task: the search waits for a pause.
        if case .loaded = search { try? await Task.sleep(for: .milliseconds(400)) }
        guard !Task.isCancelled else { return }
        if case .loaded = search {} else { search = .loading }
        do {
            let results = try await HubSearch.search(query)
            guard !Task.isCancelled else { return }
            search = .loaded(Self.sorted(results))
            if let selectedResultID, !results.contains(where: { $0.id == selectedResultID }) { self.selectedResultID = nil }
        } catch {
            guard !Task.isCancelled, !(error is CancellationError), (error as? URLError)?.code != .cancelled else { return }
            search = .failed(error.localizedDescription)
        }
    }

    private var selectedResult: HubModel? {
        guard let selectedResultID, case .loaded(let results) = search else { return nil }
        return results.first { $0.id == selectedResultID }
    }

    private var selectedVerdict: ModelCompatibility.Verdict? {
        selectedResult.map(ModelCompatibility.assess)
    }

    @ViewBuilder
    private func selectionForm(_ result: HubModel) -> some View {
        let verdict = ModelCompatibility.assess(result)
        if case .unsupported(let reason) = verdict {
            Label(reason, systemImage: "xmark.circle")
                .font(.callout)
                .foregroundStyle(.secondary)
        } else {
            Form {
                if case .unknownFamily(let families) = verdict {
                    Picker("Family", selection: $family) {
                        ForEach(families, id: \.self) { Text($0.displayName).tag($0) }
                    }
                    .onAppear { if !families.contains(family) { family = families[0] } }
                    if family == .flux2Klein { kleinVariantPicker }
                }
                TextField("Name", text: $name, prompt: Text(result.name))
                if result.gated {
                    Label("Gated: accept its terms on huggingface.co and add your token in Settings before it downloads.",
                          systemImage: "lock")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.columns)
        }
    }

    // MARK: By hand

    private var manualForm: some View {
        Form {
            Picker("Family", selection: $family) {
                ForEach(ModelFamily.allCases, id: \.self) { family in
                    Text(family.displayName).tag(family)
                }
            }
            if family == .flux2Klein { kleinVariantPicker }

            switch origin {
            case .folder:
                LabeledContent("Folder") {
                    HStack {
                        Text(folder?.path(percentEncoded: false) ?? "None")
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(folder == nil ? .secondary : .primary)
                        Button("Choose…", action: chooseFolder)
                    }
                }
            default:
                TextField("Repository", text: $repo, prompt: Text("organization/model-name"))
                repoCheckLabel
            }
            TextField("Name", text: $name, prompt: Text(suggestedName.isEmpty ? "My model" : suggestedName))
        }
        .formStyle(.grouped)
        .task(id: origin == .huggingFace ? trimmedRepo : "") { await checkRepo() }
    }

    private var kleinVariantPicker: some View {
        Picker("Variant", selection: $kleinVariant) {
            Text("From the model’s name").tag("")
            ForEach(ModelCatalog.kleinVariants, id: \.key) { variant in
                Text(variant.label).tag(variant.key)
            }
        }
    }

    /// What the hub says of the repository: a warning only, since a checkpoint can still be right
    /// for a family it does not name.
    @ViewBuilder
    private var repoCheckLabel: some View {
        switch repoCheck {
        case .none:
            EmptyView()
        case .checking:
            Label("Checking on Hugging Face…", systemImage: "ellipsis.circle")
                .foregroundStyle(.secondary)
        case .notFound:
            Label("Not found on Hugging Face (or private to another account).", systemImage: "questionmark.circle")
                .foregroundStyle(.secondary)
        case .found(.supported(let family, _)):
            Label("A \(family.displayName) checkpoint Turbo MLX runs.", systemImage: "checkmark.circle")
                .foregroundStyle(.secondary)
        case .found(.unknownFamily):
            Label("Its files fit; Hugging Face doesn't say which family it is.", systemImage: "questionmark.circle")
                .foregroundStyle(.secondary)
        case .found(.unsupported(let reason)):
            Label(reason, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        }
    }

    private func checkRepo() async {
        guard origin == .huggingFace, case .huggingFace(let repo)? = source else {
            repoCheck = .none
            return
        }
        try? await Task.sleep(for: .milliseconds(500))
        guard !Task.isCancelled else { return }
        repoCheck = .checking
        do {
            let model = try await HubSearch.model(repo)
            guard !Task.isCancelled else { return }
            guard let model else {
                repoCheck = .notFound
                return
            }
            let verdict = ModelCompatibility.assess(model)
            repoCheck = .found(verdict)
            if case .supported(let family, let variant) = verdict {
                self.family = family
                kleinVariant = variant ?? ""
            }
        } catch {
            // Offline or the hub busy: the repository may still be fine, so nothing is said.
            if !Task.isCancelled { repoCheck = .none }
        }
    }

    // MARK: Adding

    private var trimmedRepo: String { repo.trimmingCharacters(in: .whitespacesAndNewlines) }

    private var source: ModelDescriptor.Source? {
        switch origin {
        case .search:
            switch selectedVerdict {
            case .supported, .unknownFamily: return selectedResult.map { .huggingFace(repo: $0.id) }
            case .unsupported, nil: return nil
            }
        case .huggingFace:
            let parts = trimmedRepo.split(separator: "/")
            return parts.count == 2 && !trimmedRepo.contains(" ") ? .huggingFace(repo: trimmedRepo) : nil
        case .folder:
            return folder.map { .local(path: $0.path(percentEncoded: false)) }
        }
    }

    private var suggestedName: String {
        switch origin {
        case .search: selectedResult?.name ?? ""
        case .huggingFace: trimmedRepo.split(separator: "/").last.map(String.init) ?? ""
        case .folder: folder?.lastPathComponent ?? ""
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK { folder = panel.url }
    }

    private func add() {
        guard let source else { return }
        var family = family
        var variant = family == .flux2Klein && !kleinVariant.isEmpty ? kleinVariant : nil
        if origin == .search, case .supported(let found, let foundVariant) = selectedVerdict {
            family = found
            variant = foundVariant
        }
        // The sandbox forgets a folder chosen in the panel when the app quits; a bookmark keeps it.
        if origin == .folder, let folder { FolderAccess.remember(folder) }
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        app.addCustomModel(ModelDescriptor(
            name: trimmedName.isEmpty ? suggestedName : trimmedName,
            detail: "Custom \(family.displayName) checkpoint "
                + (origin == .folder ? "from a local folder." : "from Hugging Face."),
            family: family,
            source: source,
            variant: variant
        ))
        dismiss()
    }
}

/// A search result: the repository, its author and downloads, and whether Turbo MLX runs it.
private struct ResultRow: View {
    @Environment(\.openURL) private var openURL
    let result: HubModel
    let verdict: ModelCompatibility.Verdict
    let isSelected: Bool
    let select: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Button(action: select) {
                summary
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(isUnsupported)
            .accessibilityAddTraits(isSelected ? .isSelected : [])
            Label(result.downloads.formatted(.number.notation(.compactName)), systemImage: "arrow.down.circle")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .help("Downloads last month")
            Button {
                if let url = URL(string: "https://huggingface.co/\(result.id)") { openURL(url) }
            } label: {
                Image(systemName: "arrow.up.right.square")
            }
            .buttonStyle(.borderless)
            .help("View on Hugging Face")
            .accessibilityLabel("View \(result.id) on Hugging Face")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(isSelected ? Color.active.opacity(0.18) : .clear)
    }

    private var summary: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Text(result.name)
                        .fontWeight(.medium)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if result.gated {
                        Image(systemName: "lock.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .help("Gated: its terms must be accepted on Hugging Face")
                    }
                }
                Text(result.author)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                status
                    .font(.caption)
            }
            .opacity(isUnsupported ? 0.55 : 1)
            Spacer(minLength: 8)
        }
    }

    private var isUnsupported: Bool {
        if case .unsupported = verdict { true } else { false }
    }

    @ViewBuilder
    private var status: some View {
        switch verdict {
        case .supported(let family, let variant):
            let label = variant.flatMap { key in ModelCatalog.kleinVariants.first { $0.key == key }?.label }
            Label(label.map { "\(family.displayName) · \($0)" } ?? family.displayName, systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .unknownFamily:
            Label("Choose its family", systemImage: "questionmark.circle")
                .foregroundStyle(.secondary)
        case .unsupported(let reason):
            Text(reason)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .help(reason)
        }
    }
}
