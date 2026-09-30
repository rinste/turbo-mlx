import SwiftUI
import UniformTypeIdentifiers

/// Left column: model, prompt and generation settings.
struct ControlPanel: View {
    @Environment(AppModel.self) private var app
    @FocusState private var focusedBlock: PromptBlock.ID?
    /// Size, seed, outputs and memory: most images don't need them.
    @AppStorage("showsAdvancedSettings") private var showsAdvanced = false

    var body: some View {
        @Bindable var app = app
        Form {
            if let model = app.selectedModel {
                if model.family.takesReferenceImage {
                    ReferenceImageSection(settings: $app.settings)
                }
                if model.family.isUpscaler {
                    // No prompt and no format: the picture, and how much larger.
                    UpscaleSection(settings: $app.settings)
                } else {
                    PromptSection(settings: $app.settings, focus: $focusedBlock)
                    FormatSection(settings: $app.settings)
                    if model.family.media == .video {
                        ClipSection(settings: $app.settings)
                    }
                    ParametersSection(settings: $app.settings, model: model)
                }
                AdvancedSection(isExpanded: $showsAdvanced, settings: $app.settings, family: model.family)
            }
        }
        .formStyle(.grouped)
        // Air around the sections: the grouped style alone leaves them close to the edge and,
        // with a mouse, right under the scroll bar.
        .contentMargins(.leading, 6, for: .scrollContent)
        .contentMargins(.trailing, 14, for: .scrollContent)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            GenerateBar()
        }
        .onAppear { focusedBlock = app.settings.blocks.first?.id }
    }
}

// MARK: - Reset

/// Back to the model and settings of a first launch, with an empty prompt; ⌘Z brings them back.
/// Next to the model, in the tray.
private struct ResetButton: View {
    @Environment(AppModel.self) private var app
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        Button("Reset") {
            app.resetControls(undoManager: undoManager)
        }
        .help("Reset the prompt, the model and the settings to their defaults (⌘Z undoes it)")
    }
}

// MARK: - Model

/// The model the main button generates with, right above it: the picker, what the model is for,
/// and whether it's downloaded and in memory.
private struct ModelSelection: View {
    @Environment(AppModel.self) private var app
    @State private var showsAddModel = false

    var body: some View {
        @Bindable var app = app
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("Model:")
                // Upscalers, image models, then video models; within each, the names say the family
                // and the memory.
                ModelPopUp(
                    sections: [
                        section("Upscale") { $0.family.isUpscaler },
                        section("Image") { $0.family.media == .image && !$0.family.isUpscaler },
                        section("Video") { $0.family.media == .video },
                    ].filter { !$0.entries.isEmpty },
                    selection: $app.selectedModelID,
                    help: "A picture: the model also takes a reference image. Lines: it works from the prompt alone. A small arrow: not downloaded yet."
                )
                .frame(maxWidth: .infinity, alignment: .leading)
                ResetButton()
                ModelMenu(showsAddModel: $showsAddModel)
            }

            if let model = app.selectedModel {
                // A line limit, not fixedSize: this bar does not scroll, and a text sized for any
                // width asks for one character per line when the window measures its minimum
                // height, which then pushes the whole window's content past its edges.
                Text(model.detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                ModelStatusRow(model: model)
                MemoryWarning(model: model)
            }
        }
        .sheet(isPresented: $showsAddModel) {
            AddModelSheet()
        }
    }

    private func section(_ title: String, including: (ModelDescriptor) -> Bool) -> ModelPopUp.Section {
        let entries = app.models.filter(including).map { model in
            ModelPopUp.Entry(
                id: model.id, name: model.shortName != model.name ? model.shortName : model.name,
                badge: model.shortName != model.name ? model.memoryLabel : nil,
                icon: ModelIcon.image(takesPicture: model.family.takesReferenceImage, installed: app.isInstalled(model))
            )
        }
        return ModelPopUp.Section(title: title, entries: entries)
    }
}

/// The model picker: a pop-up button whose menu opens whole above it, out of the tray at the bottom
/// of the window. A standard pop-up would center the chosen model on the button, and on a short
/// screen the models below it then went off the edge, behind a scroll arrow. Each model shows its
/// icon, its name and its memory on a badge centered on the name.
private struct ModelPopUp: NSViewRepresentable {
    struct Entry {
        let id: String
        let name: String
        let badge: String?
        let icon: NSImage
    }

    struct Section {
        let title: String
        let entries: [Entry]
    }

    let sections: [Section]
    @Binding var selection: String
    let help: String

    func makeCoordinator() -> Coordinator { Coordinator(selection: $selection) }

    func makeNSView(context: Context) -> UpwardPopUpButton {
        let button = UpwardPopUpButton(frame: .zero, pullsDown: false)
        button.setContentHuggingPriority(.defaultLow, for: .horizontal)
        button.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        button.lineBreakMode = .byTruncatingTail
        button.setAccessibilityLabel("Model")
        return button
    }

    func updateNSView(_ button: UpwardPopUpButton, context: Context) {
        context.coordinator.selection = $selection
        let font = NSFont.menuFont(ofSize: NSFont.systemFontSize)
        let menu = NSMenu()
        menu.font = font
        var chosen: NSMenuItem?
        for (index, section) in sections.enumerated() {
            if index > 0 { menu.addItem(.separator()) }
            menu.addItem(.sectionHeader(title: section.title))
            for entry in section.entries {
                let item = NSMenuItem(title: entry.name, action: #selector(Coordinator.choose(_:)), keyEquivalent: "")
                item.target = context.coordinator
                item.representedObject = entry.id
                item.image = entry.icon
                if let badge = entry.badge { item.attributedTitle = Self.title(entry.name, badge: badge, font: font) }
                menu.addItem(item)
                if entry.id == selection { chosen = item }
            }
        }
        button.menu = menu
        if let chosen { button.select(chosen) }
        button.toolTip = help
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: UpwardPopUpButton, context: Context) -> CGSize? {
        CGSize(width: proposal.width ?? nsView.intrinsicContentSize.width, height: nsView.intrinsicContentSize.height)
    }

    /// The name, then the badge, lowered from the baseline (where a picture in a text stands) so
    /// its middle meets the middle of the capitals.
    static func title(_ name: String, badge: String, font: NSFont) -> NSAttributedString {
        let text = NSMutableAttributedString(string: name + "  ", attributes: [.font: font])
        let image = MemoryBadge.image(badge)
        let attachment = NSTextAttachment()
        attachment.image = image
        attachment.bounds = CGRect(x: 0, y: ((font.capHeight - image.size.height) / 2).rounded(), width: image.size.width, height: image.size.height)
        text.append(NSAttributedString(attachment: attachment))
        return text
    }

    final class Coordinator: NSObject {
        var selection: Binding<String>

        init(selection: Binding<String>) {
            self.selection = selection
        }

        @objc func choose(_ item: NSMenuItem) {
            if let id = item.representedObject as? String { selection.wrappedValue = id }
        }
    }
}

/// A pop-up button that opens its menu above itself, whole, at least as wide as the button.
final class UpwardPopUpButton: NSPopUpButton {
    override func mouseDown(with event: NSEvent) {
        guard isEnabled, let menu else { return super.mouseDown(with: event) }
        popUpAbove(menu)
    }

    override func performClick(_ sender: Any?) {
        guard isEnabled, let menu else { return super.performClick(sender) }
        popUpAbove(menu)
    }

    private func popUpAbove(_ menu: NSMenu) {
        menu.minimumWidth = bounds.width
        // The menu's top left corner, in this view's coordinates: its whole height above the button.
        let gap: CGFloat = 4
        let top = isFlipped ? -(menu.size.height + gap) : bounds.height + menu.size.height + gap
        highlight(true)
        menu.popUp(positioning: nil, at: NSPoint(x: 0, y: top), in: self)
        highlight(false)
    }
}

/// A model's icon in the picker: what it reads, a picture as well as the prompt or the prompt
/// alone, with a small arrow while it is not downloaded. One template image of a fixed width, so the
/// names line up and the menu tints it like its own symbols.
private enum ModelIcon {
    private static var cache: [String: NSImage] = [:]

    static func image(takesPicture: Bool, installed: Bool) -> NSImage {
        let key = "\(takesPicture)-\(installed)"
        if let cached = cache[key] { return cached }
        func symbol(_ name: String, size: CGFloat, weight: NSFont.Weight = .regular) -> NSImage {
            let configuration = NSImage.SymbolConfiguration(pointSize: size, weight: weight)
            return NSImage(systemSymbolName: name, accessibilityDescription: nil)?.withSymbolConfiguration(configuration) ?? NSImage()
        }
        let base = symbol(takesPicture ? "photo" : "text.alignleft", size: 13)
        let slot = max(symbol("photo", size: 13).size.width, symbol("text.alignleft", size: 13).size.width)
        let badge = symbol("arrow.down.circle.fill", size: 8, weight: .bold)
        let size = NSSize(width: slot + 4, height: max(base.size.height, 16))
        let image = NSImage(size: size, flipped: false) { rect in
            base.draw(in: NSRect(x: (slot - base.size.width) / 2, y: (rect.height - base.size.height) / 2,
                                 width: base.size.width, height: base.size.height))
            if !installed {
                // The arrow in the lower right corner, cut out of the symbol under it.
                let corner = NSRect(x: rect.maxX - badge.size.width, y: 0, width: badge.size.width, height: badge.size.height)
                NSGraphicsContext.current?.compositingOperation = .clear
                NSBezierPath(ovalIn: corner.insetBy(dx: -1.5, dy: -1.5)).fill()
                NSGraphicsContext.current?.compositingOperation = .sourceOver
                badge.draw(in: corner)
            }
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = (takesPicture ? "Takes a reference image" : "Prompt only") + (installed ? "" : ", not downloaded")
        cache[key] = image
        return image
    }
}

/// "24 GB RAM" in small white letters on a dark pill, drawn once as an image so a menu item can
/// carry it after the name. The capitals sit in the middle of the pill, with room around them.
private enum MemoryBadge {
    private static var cache: [String: NSImage] = [:]
    private static let height: CGFloat = 15

    static func image(_ text: String) -> NSImage {
        if let cached = cache[text] { return cached }
        let font = NSFont.systemFont(ofSize: 8, weight: .semibold)
        let attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white, .kern: 0.3]
        let textWidth = ceil((text as NSString).size(withAttributes: attributes).width)
        let size = NSSize(width: textWidth + 14, height: height)
        let image = NSImage(size: size, flipped: false) { rect in
            let pill = NSBezierPath(roundedRect: rect.insetBy(dx: 0.5, dy: 0.5), xRadius: rect.height / 2 - 0.5, yRadius: rect.height / 2 - 0.5)
            NSColor(white: 0.08, alpha: 0.9).setFill()
            pill.fill()
            NSColor(white: 1, alpha: 0.18).setStroke()
            pill.lineWidth = 1
            pill.stroke()
            // The baseline where the capitals' middle is the pill's (the descender lies below the
            // point the text is drawn at).
            let baseline = rect.midY - font.capHeight / 2
            (text as NSString).draw(at: NSPoint(x: 7, y: baseline + font.descender), withAttributes: attributes)
            return true
        }
        image.accessibilityDescription = text
        cache[text] = image
        return image
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
            .lineLimit(3) // in the bar too: see ModelSelection
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
    @State private var modelToTrash: ModelDescriptor?

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
                if app.downloadFiles(of: model) != nil {
                    Button("Move to Trash…") { modelToTrash = model }
                        .disabled(!app.canTrash(model))
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
        .confirmsTrashing($modelToTrash)
    }
}

// MARK: - Prompt

/// The prompt as blocks: each one a piece of the final text, renamable, removable, and put in
/// order by dragging its handle (or with its arrows).
private struct PromptSection: View {
    @Binding var settings: GenerationSettings
    var focus: FocusState<PromptBlock.ID?>.Binding

    /// The block being dragged; back to nil when it's let go, or when the drag is cancelled.
    @GestureState(resetTransaction: Transaction(animation: .snappy)) private var drag: BlockDrag? = nil
    /// Each block's height, to know where a dragged block would land.
    @State private var heights: [PromptBlock.ID: CGFloat] = [:]
    /// The block let go last: it stays above the others while it springs into place.
    @State private var settling: PromptBlock.ID?

    private static let space = NamedCoordinateSpace.named("promptBlocks")

    var body: some View {
        let reordering = drag.flatMap { Reordering(of: settings.blocks, heights: heights, drag: $0) }
        Section {
            // All the blocks in one row, so the dragged one can be drawn over the others.
            VStack(spacing: 0) {
                ForEach($settings.blocks) { $block in
                    // Read once: `block` goes through the binding, which may no longer point
                    // at this block when a closure runs, after a removal say.
                    let id = block.id
                    let index = position(of: id)
                    let isDragged = reordering?.source == index
                    let offset = reordering?.offset(at: index) ?? 0
                    let place = reordering?.place(of: index) ?? index
                    PromptBlockEditor(
                        block: $block,
                        position: index,
                        count: settings.blocks.count,
                        isDragged: isDragged,
                        showsSeparator: !isDragged && place > 0,
                        focus: focus,
                        reorder: reorderGesture(for: id),
                        onMove: { move(id, by: $0) },
                        onRemove: { remove(id) }
                    )
                    .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { heights[id] = $0 }
                    .offset(y: offset)
                    // The dragged block sticks to the pointer; the others slide out of its way.
                    .animation(isDragged ? nil : .snappy, value: offset)
                    .zIndex(isDragged ? 2 : id == settling ? 1 : 0)
                }
            }
            .coordinateSpace(Self.space)
        } header: {
            Text("Prompt")
        } footer: {
            VStack(alignment: .leading, spacing: 8) {
                Button("Add Prompt Block", systemImage: "plus") { addBlock() }
                    .labelStyle(.titleAndIcon)
                    .help("Add a block at the end: the blocks are joined, in order, into one prompt")
                if settings.blocks.count > 1 {
                    Text("The blocks are joined, in order, into one prompt.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func position(of id: PromptBlock.ID) -> Int {
        settings.blocks.firstIndex { $0.id == id } ?? 0
    }

    private func addBlock() {
        let block = PromptBlock(name: PromptBlock.defaultName(among: settings.blocks))
        withAnimation(.snappy) { settings.blocks.append(block) }
        focus.wrappedValue = block.id
    }

    /// `offset` -1 moves the block up, +1 down.
    private func move(_ id: PromptBlock.ID, by offset: Int) {
        guard let index = settings.blocks.firstIndex(where: { $0.id == id }) else { return }
        let target = index + offset
        guard settings.blocks.indices.contains(target) else { return }
        withAnimation(.snappy) { settings.blocks.swapAt(index, target) }
    }

    private func remove(_ id: PromptBlock.ID) {
        guard settings.blocks.count > 1, let index = settings.blocks.firstIndex(where: { $0.id == id }) else { return }
        withAnimation(.snappy) { _ = settings.blocks.remove(at: index) }
        if focus.wrappedValue == id {
            focus.wrappedValue = settings.blocks[min(index, settings.blocks.count - 1)].id
        }
    }

    /// The blocks make room while one is dragged; the list itself changes when it's let go.
    private func reorderGesture(for id: PromptBlock.ID) -> some Gesture {
        DragGesture(minimumDistance: 2, coordinateSpace: Self.space)
            .updating($drag) { value, state, _ in
                state = BlockDrag(id: id, translation: value.translation.height)
            }
            .onEnded { value in
                settling = id
                let drop = BlockDrag(id: id, translation: value.translation.height)
                guard let reordering = Reordering(of: settings.blocks, heights: heights, drag: drop),
                      reordering.destination != reordering.source
                else { return }
                var blocks = settings.blocks
                blocks.insert(blocks.remove(at: reordering.source), at: reordering.destination)
                withAnimation(.snappy) { settings.blocks = blocks }
            }
    }
}

private struct PromptBlockEditor<Reorder: Gesture>: View {
    @Environment(AppModel.self) private var app
    @Binding var block: PromptBlock
    /// Index in the list, and how many blocks there are: the arrows stop at the ends.
    let position: Int
    let count: Int
    /// Lifted above the others while its handle is dragged.
    let isDragged: Bool
    let showsSeparator: Bool
    var focus: FocusState<PromptBlock.ID?>.Binding
    /// The drag that starts on the handle.
    let reorder: Reorder
    let onMove: (Int) -> Void
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 2) {
                if count > 1 {
                    Image(systemName: "line.3.horizontal")
                        .foregroundStyle(.tertiary)
                        .frame(width: 20, height: 20)
                        .contentShape(Rectangle())
                        .pointerStyle(isDragged ? .grabActive : .grabIdle)
                        .gesture(reorder)
                        .padding(.trailing, 4)
                        .help("Drag to move the block")
                        .accessibilityHidden(true)
                }
                // The name is the block's title, renamed in place; Return goes on to the text.
                TextField("Name", text: $block.name)
                    .labelsHidden()
                    .textFieldStyle(.plain)
                    .multilineTextAlignment(.leading)
                    .font(.headline)
                    .onSubmit { focus.wrappedValue = block.id }
                    .help("Click to rename the block")
                Button {
                    onMove(-1)
                } label: {
                    Image(systemName: "chevron.up")
                }
                .disabled(position == 0)
                .help("Move up")
                Button {
                    onMove(1)
                } label: {
                    Image(systemName: "chevron.down")
                }
                .disabled(position >= count - 1)
                .help("Move down")
                Button {
                    onRemove()
                } label: {
                    Image(systemName: "xmark.circle")
                }
                .disabled(count <= 1)
                .help("Remove this block")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)

            BlockTextEditor(
                text: $block.text,
                height: $block.height,
                placeholder: placeholder,
                focus: focus,
                id: block.id
            )
        }
        .contextMenu {
            Button("Move Up") { onMove(-1) }
                .disabled(position == 0)
            Button("Move Down") { onMove(1) }
                .disabled(position >= count - 1)
            Divider()
            Button("Remove Block", role: .destructive) { onRemove() }
                .disabled(count <= 1)
        }
        // No bottom padding: the resize handle under the text is the block's lower margin.
        .padding(.top, 6)
        .background {
            if isDragged {
                RoundedRectangle(cornerRadius: 8)
                    .fill(.regularMaterial)
                    .shadow(color: .black.opacity(0.2), radius: 6, y: 2)
                    .padding(.horizontal, -6)
            }
        }
        .overlay(alignment: .top) {
            if showsSeparator { Divider() }
        }
    }

    private var placeholder: String {
        let video = app.selectedModel?.family.media == .video
        switch block.name {
        case PromptBlock.subjectName:
            return video
                ? "What happens: who or what, where, what they do, and what we hear."
                : "What the image shows: who or what, where, doing what. Put any text to render in quotes."
        case PromptBlock.styleName:
            return video
                ? "How it looks and moves: the style, the light, the camera."
                : "How it looks: the medium or style, the light, the colors, the lens."
        default:
            if position > 0 { return "A piece of the prompt: the style, the lighting, a detail. It follows the block above." }
            return video
                ? "Describe the clip: the action, the camera, the light, the sound."
                : "Describe the image: subject, style, light, colors. Put any text to render in quotes."
        }
    }
}

/// A block's text, as tall as the user makes it with the handle along its bottom edge.
private struct BlockTextEditor: View {
    @Binding var text: String
    /// Nil until the block is resized.
    @Binding var height: Double?
    let placeholder: String
    var focus: FocusState<PromptBlock.ID?>.Binding
    let id: PromptBlock.ID

    /// How far the handle has been dragged; the new height is kept when it's let go.
    @GestureState private var stretch: CGFloat = 0

    private static let defaultHeight: CGFloat = 110
    /// From two lines to about thirty.
    private static let heights: ClosedRange<CGFloat> = 44...600

    var body: some View {
        VStack(spacing: 0) {
            TextEditor(text: $text)
                .scrollContentBackground(.hidden)
                .frame(height: textHeight(stretchedBy: stretch))
                .focused(focus, equals: id)
                .overlay(alignment: .topLeading) {
                    if text.isEmpty {
                        Text(placeholder)
                            .foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
                // Larger than the controls around it: the prompt is what gets read and written here.
                .font(.system(size: 15))

            Capsule()
                .fill(.tertiary)
                .frame(width: 32, height: 4)
                .frame(maxWidth: .infinity, minHeight: 14)
                .contentShape(Rectangle())
                .pointerStyle(.frameResize(position: .bottom))
                .gesture(
                    // In global coordinates: the handle itself moves down as the text grows.
                    DragGesture(minimumDistance: 1, coordinateSpace: .global)
                        .updating($stretch) { value, state, _ in
                            state = value.translation.height
                        }
                        .onEnded { value in
                            height = Double(textHeight(stretchedBy: value.translation.height))
                        }
                )
                .help("Drag to make the block taller or shorter")
                .accessibilityHidden(true)
        }
    }

    private func textHeight(stretchedBy amount: CGFloat) -> CGFloat {
        let start = height.map { CGFloat($0) } ?? Self.defaultHeight
        return min(max(start + amount, Self.heights.lowerBound), Self.heights.upperBound)
    }
}

/// A block dragged by its handle, and how far the pointer has moved since.
private struct BlockDrag {
    let id: PromptBlock.ID
    let translation: CGFloat
}

/// Where the blocks are drawn while one of them is dragged: that one follows the pointer, kept
/// within the list, and the others slide out of the way of the place it would land in.
private struct Reordering {
    /// The dragged block's index, and the index it would land at.
    let source: Int
    let destination: Int
    private let translation: CGFloat
    private let draggedHeight: CGFloat

    init?(of blocks: [PromptBlock], heights: [PromptBlock.ID: CGFloat], drag: BlockDrag) {
        guard let source = blocks.firstIndex(where: { $0.id == drag.id }) else { return nil }
        let height: (PromptBlock) -> CGFloat = { heights[$0.id] ?? 0 }
        let others = blocks.filter { $0.id != drag.id }
        let top: CGFloat = blocks[..<source].reduce(0) { $0 + height($1) }
        let othersHeight: CGFloat = others.reduce(0) { $0 + height($1) }
        let translation = min(max(drag.translation, -top), othersHeight - top)
        // It lands in the place whose top is nearest its own: past the middle of a neighbor, it
        // takes the neighbor's place.
        var destination = 0
        var placeTop: CGFloat = 0
        for other in others {
            guard top + translation > placeTop + height(other) / 2 else { break }
            destination += 1
            placeTop += height(other)
        }
        self.source = source
        self.destination = destination
        self.translation = translation
        draggedHeight = height(blocks[source])
    }

    /// How far the block at `index` is drawn from its place in the list.
    func offset(at index: Int) -> CGFloat {
        if index == source { return translation }
        if index < source, index >= destination { return draggedHeight }
        if index > source, index <= destination { return -draggedHeight }
        return 0
    }

    /// Where the block at `index` is drawn, counting from the top.
    func place(of index: Int) -> Int {
        if index == source { return destination }
        if index < source, index >= destination { return index + 1 }
        if index > source, index <= destination { return index - 1 }
        return index
    }
}

// MARK: - Reference image

/// The image a clip starts from, or an image is made from: dropped from Finder or from the history
/// strip, chosen with the open panel, or picked among the generated images. A clip takes it as its
/// first frame and the prompt says what happens next; FLUX.2 Klein and Qwen-Image Edit change it as
/// the prompt says.
private struct ReferenceImageSection: View {
    @Environment(AppModel.self) private var app
    @Binding var settings: GenerationSettings
    @State private var isTargeted = false
    @State private var showsHistory = false
    @State private var failure: String?

    private var isVideo: Bool { app.selectedModel?.family.media == .video }
    private var isUpscaler: Bool { app.selectedModel?.family.isUpscaler == true }

    private var caption: String {
        if isVideo { return "Optional. It is fitted to the clip’s size, cropped from the middle." }
        if isUpscaler { return "Needed. It keeps its proportions; a picture larger than 2048 pixels is reduced to that first." }
        if app.selectedModel?.family.requiresReferenceImage == true {
            return "Needed: the picture to edit. It keeps its own proportions, at about the new image’s size; the new image takes the format below."
        }
        return "Optional. It keeps its own proportions, at most about a megapixel; the new image takes the format below."
    }

    var body: some View {
        Section {
            HStack(alignment: .center, spacing: 12) {
                preview
                VStack(alignment: .leading, spacing: 8) {
                    Text(prompt)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 8) {
                        Button("Choose…") { choose() }
                            .help("Choose an image file")
                        Button("From History…") { showsHistory = true }
                            .help("Pick one of the images generated here, or a clip’s first frame")
                            .disabled(app.history.items.isEmpty)
                            .popover(isPresented: $showsHistory, arrowEdge: .trailing) {
                                HistoryImagePicker { item in
                                    showsHistory = false
                                    use(app.url(for: item))
                                }
                            }
                        if settings.referenceImage != nil {
                            Button("Remove", role: .destructive) {
                                withAnimation(.snappy) { settings.referenceImage = nil }
                            }
                        }
                    }
                    .controlSize(.small)
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 2)
            .contentShape(Rectangle())
            .onDrop(of: [.fileURL, .image, .movie], isTargeted: $isTargeted) { providers in accept(providers) }
            .overlay {
                if isTargeted {
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color.active, lineWidth: 2)
                        .padding(-6)
                }
            }
        } header: {
            Text(isUpscaler ? "Picture to Upscale" : "Reference Image")
        } footer: {
            if let failure {
                Text(failure).font(.caption).foregroundStyle(.red)
            } else {
                Text(caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var prompt: String {
        switch (settings.referenceImage == nil, isVideo, isUpscaler) {
        case (true, true, _): "Drop an image here, or pick one: the clip starts from it."
        case (true, _, true): "Drop an image here, or pick one: it comes out larger, with sharper detail."
        case (true, _, _): "Drop an image here, or pick one: the prompt says what to change in it."
        case (false, true, _): "The clip starts from this image."
        case (false, _, true): "This picture is upscaled."
        case (false, _, _): "The new image is made from this one, as the prompt says."
        }
    }

    @ViewBuilder
    private var preview: some View {
        let shape = RoundedRectangle(cornerRadius: 6)
        if let name = settings.referenceImage {
            FileImage(url: HistoryStore.referenceURL(name), maxPixelSize: 320)
                .aspectRatio(contentMode: .fill)
                .frame(width: 96, height: 72)
                .clipShape(shape)
                .overlay { shape.strokeBorder(Color.primary.opacity(0.12)) }
                .help("Drop another image to replace it")
        } else {
            shape
                .strokeBorder(.tertiary, style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                .frame(width: 96, height: 72)
                .overlay {
                    Image(systemName: "photo.badge.plus")
                        .font(.title2)
                        .foregroundStyle(.tertiary)
                }
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.message = isUpscaler ? "Choose the picture to upscale" : isVideo ? "Choose the image the clip starts from" : "Choose the reference image"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        use(url)
    }

    private func use(_ url: URL, then cleanUp: (() -> Void)? = nil) {
        Task {
            defer { cleanUp?() }
            await importing { try await HistoryStore.importReference(from: url) }
        }
    }

    private func importing(_ load: () async throws -> String) async {
        do {
            let name = try await load()
            failure = nil
            withAnimation(.snappy) { settings.referenceImage = name }
        } catch {
            failure = error.localizedDescription
        }
    }

    /// A file from Finder; an image from the history, a browser or Photos; a clip from the history,
    /// which arrives as a copy that lasts as long as the callback and is moved aside first.
    private func accept(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        if provider.canLoadObject(ofClass: URL.self) {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in use(url) }
            }
            return true
        }
        if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data else { return }
                Task { @MainActor in await importing { try HistoryStore.importReference(data: data) } }
            }
            return true
        }
        if provider.hasItemConformingToTypeIdentifier(UTType.movie.identifier) {
            _ = provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { url, _ in
                guard let url else { return }
                let copy = FileManager.default.temporaryDirectory
                    .appending(path: UUID().uuidString).appendingPathExtension(url.pathExtension)
                guard (try? FileManager.default.copyItem(at: url, to: copy)) != nil else { return }
                Task { @MainActor in use(copy) { try? FileManager.default.removeItem(at: copy) } }
            }
            return true
        }
        return false
    }
}

/// The generated images and clips (by their first frame), newest first, to pick a reference from.
private struct HistoryImagePicker: View {
    @Environment(AppModel.self) private var app
    let pick: (HistoryItem) -> Void

    var body: some View {
        let items = app.history.items
        let rows = (items.count + 3) / 4
        ScrollView {
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(88), spacing: 8), count: 4), spacing: 8) {
                ForEach(items) { item in
                    Button { pick(item) } label: {
                        FileImage(url: app.posterURL(for: item), maxPixelSize: 320)
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 88, height: 88)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay { RoundedRectangle(cornerRadius: 6).strokeBorder(Color.primary.opacity(0.1)) }
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
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(item.caption)
                }
            }
            .padding(12)
        }
        .frame(width: 4 * 88 + 3 * 8 + 24, height: min(360, CGFloat(rows) * 96 - 8 + 24))
    }
}

// MARK: - Clip

/// How long the clip runs and at what frame rate (the frames are 8k + 1, nearest the duration).
private struct ClipSection: View {
    @Binding var settings: GenerationSettings

    var body: some View {
        Section("Clip") {
            LabeledContent {
                HStack(spacing: 6) {
                    Slider(value: $settings.videoSeconds, in: GenerationSettings.videoDurations, step: 1)
                    Text(verbatim: "\(Int(settings.videoSeconds)) s")
                        .monospacedDigit()
                        .frame(width: 34, alignment: .trailing)
                }
            } label: {
                Text("Duration")
                    .help("\(settings.videoFrames) frames. Longer clips take longer and need more memory.")
            }
            Picker("Frame rate", selection: $settings.videoFrameRate) {
                ForEach(GenerationSettings.videoFrameRates, id: \.self) { rate in
                    Text(verbatim: "\(rate) fps").tag(rate)
                }
            }
            .pickerStyle(.segmented)
            // Beyond about five seconds at 24 fps and the default resolution (768 × 512 or 832 × 512).
            if settings.videoSize.megapixels * Double(settings.videoFrames) > 55 {
                Label("Longer and larger clips take much more time and memory.", systemImage: "tortoise")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
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
                                .fill(isSelected ? Color.active.opacity(0.12) : .clear)
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("\(aspect.rawValue) · \(aspect.usage)")
                }
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
                    SliderIcon("hare", help: "Fewer steps: faster", label: "Faster")
                    Slider(
                        value: Binding(
                            get: { Double(min(max(settings.steps, range.lowerBound), range.upperBound)) },
                            set: { settings.steps = Int($0.rounded()) }
                        ),
                        in: Double(range.lowerBound)...Double(range.upperBound),
                        step: 1
                    )
                    SliderIcon("tortoise", help: "More steps: slower, sometimes more refined", label: "Slower")
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
                        // Low guidance lets the model interpret; high guidance sticks to the words.
                        SliderIcon("wand.and.stars", help: "Lower: a freer interpretation of the prompt", label: "Freer")
                        Slider(value: $settings.guidance, in: GenerationSettings.guidanceRange, step: 0.5)
                        SliderIcon("text.quote", help: "Higher: follows the prompt more literally", label: "Stricter")
                        Text(Format.guidance(settings.guidance))
                            .monospacedDigit()
                            .frame(width: 30, alignment: .trailing)
                    }
                } label: {
                    Text("Guidance")
                        .help(model.family == .qwenImage || model.family == .qwenImageEdit
                              ? "How strictly to follow the prompt. 4 is the recommended value."
                              : "How strictly to follow the prompt. 1 = off; higher values double the time of each step.")
                }
            }
        }
    }
}

// MARK: - Upscale

/// An upscaler's settings: how many times larger, how much the picture is softened first, and the
/// size that makes.
private struct UpscaleSection: View {
    @Binding var settings: GenerationSettings

    var body: some View {
        Section("Upscale") {
            Picker("Scale", selection: $settings.upscale) {
                ForEach(GenerationSettings.upscaleFactors, id: \.self) { factor in
                    Text(verbatim: "\(Int(factor))×").tag(factor)
                }
            }
            .pickerStyle(.segmented)
            .help("How many times larger each side of the picture becomes")

            LabeledContent {
                HStack(spacing: 6) {
                    SliderIcon("circle.grid.3x3", help: "None: the picture’s detail as it is", label: "Sharper")
                    Slider(value: $settings.softness, in: 0...1, step: 0.1)
                    SliderIcon("drop", help: "More: the picture is reduced first, which smooths noise and harsh edges", label: "Softer")
                    Text(verbatim: "\(Int((settings.softness * 100).rounded()))%")
                        .monospacedDigit()
                        .frame(width: 38, alignment: .trailing)
                }
            } label: {
                Text("Softness")
                    .help("0% keeps the picture’s own detail. Higher values shrink it first (up to 8 times), for smoother results from a noisy, compressed or over-sharpened picture; 50% is a good start.")
            }

            if let size = settings.upscaledSize {
                LabeledContent("Image") {
                    Text(verbatim: "\(size.width) × \(size.height) px · \(size.megapixels.formatted(.number.precision(.fractionLength(1)))) MP")
                        .monospacedDigit()
                }
                if size.megapixels > Upscale.maxMegapixels {
                    Label("Too large: at most 4096 × 4096 pixels. Choose a smaller scale.", systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.red)
                } else if size.megapixels > 9 {
                    Label("Large results take minutes and more memory: about 19 GB at 4096 × 4096.", systemImage: "tortoise")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
    }
}

/// A symbol at one end of a slider, the same width for every slider so they line up.
private struct SliderIcon: View {
    let systemImage: String
    let help: String
    let label: String

    init(_ systemImage: String, help: String, label: String) {
        self.systemImage = systemImage
        self.help = help
        self.label = label
    }

    var body: some View {
        Image(systemName: systemImage)
            .foregroundStyle(.secondary)
            .frame(width: 24)
            .help(help)
            .accessibilityLabel(label)
    }
}

// MARK: - Advanced

/// The settings most images don't need, in one box under a row that shows or hides them. Folded,
/// the row still gives the image size and a fixed seed, which change the image out of sight.
private struct AdvancedSection: View {
    @Binding var isExpanded: Bool
    @Binding var settings: GenerationSettings
    let family: ModelFamily

    var body: some View {
        Section {
            Button {
                withAnimation(.snappy) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Text("Advanced")
                    Spacer()
                    if !isExpanded {
                        Text(summary)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Image(systemName: "chevron.right")
                        .imageScale(.small)
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .accessibilityHidden(true)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")

            if isExpanded {
                // An upscale's size is its picture's, times the scale.
                if !family.isUpscaler {
                    SizeRows(settings: $settings, family: family)
                }
                OutputRows(settings: $settings, family: family)
                MemoryRows(settings: $settings, family: family)
            }
        }
    }

    private var summary: String {
        let size = settings.size(for: family)
        let dimensions = "\(size.width) × \(size.height) px"
        return settings.randomSeed ? dimensions : "\(dimensions) · seed \(settings.seed)"
    }
}

/// The image's size: a resolution for the aspect ratio above, or a width and height of its own,
/// sides rounded to the family's multiple. A clip's sides are multiples of 64, from its own,
/// smaller resolutions.
private struct SizeRows: View {
    @Binding var settings: GenerationSettings
    let family: ModelFamily

    private var video: Bool { family.media == .video }

    var body: some View {
        Picker("Resolution", selection: video ? $settings.videoResolution : $settings.resolution) {
            ForEach(video ? GenerationSettings.videoResolutions : GenerationSettings.resolutions, id: \.self) { resolution in
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

        let size = settings.size(for: family)
        LabeledContent(video ? "Clip" : "Image") {
            Text(verbatim: video
                ? "\(size.width) × \(size.height) px · \(settings.videoFrames) frames"
                : "\(size.width) × \(size.height) px · \(size.megapixels.formatted(.number.precision(.fractionLength(1)))) MP")
                .monospacedDigit()
        }
        // A clip's warning is under its duration, where its frames count too.
        if !video, size.megapixels > 2.5 {
            Label("High resolutions take much more time and memory.", systemImage: "tortoise")
                .font(.caption)
                .foregroundStyle(.orange)
        }
    }
}

/// Which images one click makes: their seeds, how many, and the background.
private struct OutputRows: View {
    @Binding var settings: GenerationSettings
    let family: ModelFamily

    var body: some View {
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

        if family.producesAlpha {
            Picker("Background", selection: $settings.transparentBackground) {
                Text("Transparent").tag(true)
                Text("White").tag(false)
            }
            .pickerStyle(.segmented)
        }
    }
}

// MARK: - Memory

private struct MemoryRows: View {
    @Environment(AppModel.self) private var app
    @Binding var settings: GenerationSettings
    let family: ModelFamily

    private var saveMemoryDescription: String {
        let base = switch family {
        case .ming:
            "Frees the text encoder (about 12 GB) once the prompt is read and decodes the image in tiles: about 15 GB instead of 35 GB at 1024 px. A prompt that was not in the queue yet loads it again."
        case .qwenImage:
            "Frees the text encoder (about 14 GB) once the prompt is read and decodes the image in tiles. A prompt that was not in the queue yet loads it again."
        case .qwenImageEdit:
            "Frees the text encoder (about 15 GB) once the prompt and the picture are read, and decodes the image in tiles. Each new prompt or picture loads it again."
        case .zImageTurbo, .flux2Klein:
            "Keeps less in memory and, where it doesn’t affect the image, decodes it in tiles."
        case .senseNova:
            "Frees the half of the model that reads the prompt (about 6 GB) once it is read: about 6 GB instead of 11 GB. A prompt that was not in the queue yet loads it again, in place of the other half."
        case .ltx2:
            "Frees Gemma and the text connector (about 14 GB) once the prompt is read, and the transformer before the clip is decoded. A prompt that was not in the queue yet loads them again."
        case .seedVR2:
            "SeedVR2 always reads and writes the picture in tiles."
        }
        return base + " On by default below 64 GB of memory."
    }

    var body: some View {
        // SeedVR2 already works in tiles: the switch would change nothing.
        if !family.isUpscaler {
            Toggle(isOn: $settings.lowMemory) {
                Text("Save memory")
                Text(saveMemoryDescription)
            }
        }
        if app.backend.loadedModelPath != nil {
            LabeledContent {
                Button("Free Memory") { app.freeMemory() }
                    .disabled(app.activeJob != nil)
            } label: {
                Text("Model loaded")
                Text("It stays in memory so the next image starts right away.")
            }
        }
    }
}

// MARK: - Generate

/// The model and the main action, under the settings, on a darker tray. The action adapts to what
/// is missing: engine, model, then generation, with how long that should take.
private struct GenerateBar: View {
    @Environment(AppModel.self) private var app
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        VStack(spacing: 10) {
            ModelSelection()
                .padding(.bottom, 4)

            if let job = app.activeJob {
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(job.statusLabel)
                            Spacer()
                            TimeLeft(job: job)
                                .foregroundStyle(.secondary)
                        }
                        .font(.caption.monospacedDigit())
                        // Twice a second: an upscale's bar follows the time.
                        TimelineView(.periodic(from: .now, by: 0.5)) { context in
                            if let fraction = job.fraction(at: context.date) {
                                ProgressView(value: fraction)
                            } else {
                                ProgressView().progressViewStyle(.linear)
                            }
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
                .buttonStyle(.primaryAction)

            if let caption {
                Text(caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(14)
        // Darker than the settings it follows: the model and the button are a tray of their own.
        .background {
            Rectangle()
                .fill(.bar)
                .overlay(Color.black.opacity(colorScheme == .dark ? 0.32 : 0.07))
        }
        .overlay(alignment: .top) { Divider() }
    }

    @ViewBuilder
    private var primaryButton: some View {
        let model = app.selectedModel
        switch app.blocker {
        case .modelNotDownloaded:
            let size = model?.sizeBytes.map { " · \(Format.bytes($0))" } ?? ""
            // Nothing to generate yet: a prompt, or for an upscaler its picture.
            let isEmpty = model?.family.isUpscaler == true ? !app.hasReferenceImage : app.settings.trimmedPrompt.isEmpty
            if isEmpty {
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
        case .noModel, .missingReference, .emptyPrompt, .upscaleTooLarge, .notEnoughMemory:
            wideButton(generateTitle, systemImage: generateSymbol, note: note) {}
                .disabled(true)
        case nil:
            wideButton(generateTitle, systemImage: generateSymbol, note: note) { app.generate() }
                .help((app.isBusy
                       ? "Queue these settings: they start when the ones ahead are done (⌘↩)."
                       : "Generate (⌘↩).") + (estimateSource.map { " " + $0 } ?? ""))
        }
    }

    /// "~4 min · ⌘↩": how long the button's work should take, then its shortcut.
    private var note: String {
        [estimate.map { Format.estimate($0.seconds) }, "⌘↩"].compactMap(\.self).joined(separator: " · ")
    }

    private var estimateSource: String? {
        guard let estimate else { return nil }
        return estimate.isFromHistory
            ? "The time is judged from this Mac’s earlier generations with this model."
            : "The time is a first guess, from this model’s speed on an M1 Max; it follows this Mac once it has used the model."
    }

    /// With the settings as they are now, for all the images or clips the button would queue.
    private var estimate: TimeEstimate.Result? {
        guard let model = app.selectedModel else { return nil }
        let settings = app.settings
        return TimeEstimate.estimate(
            model: model,
            size: settings.size(for: model.family),
            steps: min(max(settings.steps, model.stepRange.lowerBound), model.stepRange.upperBound),
            guidance: settings.guidance,
            frames: model.family.media == .video ? settings.videoFrames : nil,
            reference: model.family.takesReferenceImage ? settings.referenceImage : nil,
            lowMemory: settings.lowMemory,
            count: settings.batchCount,
            // A model the queue is using will be in memory by then.
            isLoaded: app.isLoaded(model) || (app.queue.last ?? app.activeJob)?.model.id == model.id,
            history: app.history.items,
            models: app.models
        )
    }

    /// Prompt, model and settings are captured now, so the user can keep editing while it waits.
    private var generateTitle: String {
        let count = app.settings.batchCount
        if app.isBusy { return count > 1 ? "Add \(count) to Queue" : "Add to Queue" }
        if app.selectedModel?.family.isUpscaler == true { return count > 1 ? "Upscale \(count) Times" : "Upscale" }
        let things = app.selectedModel?.family.media == .video ? "Clips" : "Images"
        return count > 1 ? "Generate \(count) \(things)" : "Generate"
    }

    private var generateSymbol: String {
        app.isBusy ? "text.badge.plus" : "sparkles"
    }

    private var caption: String? {
        switch app.blocker {
        case .modelNotDownloaded: "Models download once and stay on this Mac."
        case .missingReference where app.selectedModel?.family.isUpscaler == true: "Add the picture to upscale."
        case .some(let blocker): blocker.hint
        case nil:
            switch app.pendingJobs.count {
            case 0: nil
            case 1: "Starts when the one in progress is done"
            case let ahead: "Starts after the \(ahead) ahead of it in the queue"
            }
        }
    }

    /// `note` follows the title, lighter, e.g. "Generate (~4 min · ⌘↩)".
    private func wideButton(
        _ title: String,
        systemImage: String,
        note: String? = nil,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label {
                if let note {
                    Text("\(title) \(Text(verbatim: "(\(note))").fontWeight(.regular).foregroundStyle(.secondary))")
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
