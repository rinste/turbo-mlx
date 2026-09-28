import AppKit
import SwiftUI

enum WindowID {
    static let main = "main"
    static let log = "log"
    static let acknowledgements = "acknowledgements"
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()

    func applicationWillFinishLaunching(_ notification: Notification) {
        // Writing to a worker that just died must fail with an error, not kill the app.
        signal(SIGPIPE, SIG_IGN)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let downloading = !model.downloads.active.isEmpty
        guard model.isBusy || downloading else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "Quit Turbo MLX?"
        let clip = model.activeJob?.model.family.media == .video
        alert.informativeText = model.isBusy
            ? (clip ? "A clip is being generated and will be stopped." : "An image is being generated and will be stopped.")
            : "A download is in progress. It will resume where it left off next time you open the app."
        alert.addButton(withTitle: "Quit")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn ? .terminateNow : .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) {
        model.shutdown()
    }
}

@main
struct TurboMLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("Turbo MLX", id: WindowID.main) {
            ContentView()
                .environment(delegate.model)
                .frame(minWidth: 980, minHeight: 660)
                .task { await delegate.model.start() }
        }
        .defaultSize(width: 1320, height: 860)
        .commands {
            GenerationCommands(model: delegate.model)
            ZoomCommands()
        }

        Window("Engine Log", id: WindowID.log) {
            LogView()
                .environment(delegate.model)
                .frame(minWidth: 480, minHeight: 280)
        }
        .defaultSize(width: 780, height: 480)
        .keyboardShortcut("l", modifiers: [.command, .option])

        Window("Acknowledgements", id: WindowID.acknowledgements) {
            AcknowledgementsView()
                .frame(minWidth: 480, minHeight: 320)
        }
        .defaultSize(width: 640, height: 640)

        Settings {
            SettingsView()
                .environment(delegate.model)
        }
    }
}

private struct GenerationCommands: Commands {
    let model: AppModel

    var body: some Commands {
        CommandGroup(after: .appInfo) {
            Button("Check for Updates…") { model.updater.checkForUpdates() }
                .disabled(!model.updater.canCheckForUpdates)
        }
        CommandGroup(replacing: .newItem) {}

        CommandMenu("Generate") {
            Button(model.isBusy ? "Add to Queue" : "Generate") { model.generate() }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(model.blocker != nil)
            Button("Stop") {
                if let job = model.activeJob { model.cancel(job) }
            }
            .keyboardShortcut(".", modifiers: .command)
            .disabled(model.activeJob == nil)
            Button("Clear Queue") { model.cancelAll() }
                .disabled(!model.isBusy)

            Divider()

            let item = model.displayedItem
            Button(item?.kind == .video ? "Copy Video" : "Copy Image") {
                if let item { NSPasteboard.general.copyItem(item, at: model.url(for: item)) }
            }
            .keyboardShortcut("c", modifiers: [.command, .shift])
            .disabled(item == nil)
            Button("Show in Finder") {
                if let item { NSWorkspace.shared.activateFileViewerSelecting([model.url(for: item)]) }
            }
            .disabled(item == nil)

            Divider()

            // As in the strip, by date or in the custom order: left, right, the queue last. Each loads
            // its settings.
            Button("Previous Image") { model.moveSelection(by: 1) }
                .keyboardShortcut("[", modifiers: .command)
                .disabled(model.history.items.isEmpty && !model.isBusy)
            Button("Next Image") { model.moveSelection(by: -1) }
                .keyboardShortcut("]", modifiers: .command)
                .disabled(model.history.items.isEmpty && !model.isBusy)
        }
    }
}

/// View menu, with Preview's shortcuts.
private struct ZoomCommands: Commands {
    @FocusedValue(\.imageZoom) private var zoom

    var body: some Commands {
        CommandGroup(after: .toolbar) {
            Button("Actual Size") { zoom?.zoom(to: 1) }
                .keyboardShortcut("0")
                .disabled(zoom == nil)
            Button("Zoom to Fit") { zoom?.zoomToFit() }
                .keyboardShortcut("9")
                .disabled(zoom?.isZoomedIn != true)
            Button("Zoom In") { zoom?.zoomIn() }
                .keyboardShortcut("+")
                .disabled(zoom?.canZoomIn != true)
            Button("Zoom Out") { zoom?.zoomOut() }
                .keyboardShortcut("-")
                .disabled(zoom?.isZoomedIn != true)
        }
    }
}
