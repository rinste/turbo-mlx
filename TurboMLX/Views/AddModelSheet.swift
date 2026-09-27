import SwiftUI

/// Adds a Ming-Image checkpoint that is not in the built-in catalog.
struct AddModelSheet: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    private enum Origin: Hashable {
        case huggingFace
        case folder
    }

    @State private var origin = Origin.huggingFace
    @State private var family = ModelFamily.ming
    /// FLUX.2 Klein registry key; empty means "infer from the name".
    @State private var kleinVariant = ""
    @State private var repo = ""
    @State private var folder: URL?
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Add a Model")
                    .font(.title3.bold())
                Text("A checkpoint in mflux format (saved with mflux-save, or pre-quantized): a Hugging Face repository or a local folder.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Form {
                Picker("Family", selection: $family) {
                    ForEach(ModelFamily.allCases, id: \.self) { family in
                        Text(family.displayName).tag(family)
                    }
                }
                if family == .flux2Klein {
                    Picker("Variant", selection: $kleinVariant) {
                        Text("From the model’s name").tag("")
                        ForEach(ModelCatalog.kleinVariants, id: \.key) { variant in
                            Text(variant.label).tag(variant.key)
                        }
                    }
                }
                Picker("Source", selection: $origin) {
                    Text("Hugging Face").tag(Origin.huggingFace)
                    Text("Local folder").tag(Origin.folder)
                }
                .pickerStyle(.segmented)

                switch origin {
                case .huggingFace:
                    TextField("Repository", text: $repo, prompt: Text("organization/model-name"))
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
                }
                TextField("Name", text: $name, prompt: Text(suggestedName.isEmpty ? "My model" : suggestedName))
            }
            .formStyle(.grouped)

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
        .frame(width: 480)
    }

    private var trimmedRepo: String { repo.trimmingCharacters(in: .whitespacesAndNewlines) }

    private var source: ModelDescriptor.Source? {
        switch origin {
        case .huggingFace:
            let parts = trimmedRepo.split(separator: "/")
            return parts.count == 2 && !trimmedRepo.contains(" ") ? .huggingFace(repo: trimmedRepo) : nil
        case .folder:
            return folder.map { .local(path: $0.path(percentEncoded: false)) }
        }
    }

    private var suggestedName: String {
        switch origin {
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
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        app.addCustomModel(ModelDescriptor(
            name: trimmedName.isEmpty ? suggestedName : trimmedName,
            detail: "Custom \(family.displayName) checkpoint "
                + (origin == .huggingFace ? "from Hugging Face." : "from a local folder."),
            family: family,
            source: source,
            variant: family == .flux2Klein && !kleinVariant.isEmpty ? kleinVariant : nil
        ))
        dismiss()
    }
}
