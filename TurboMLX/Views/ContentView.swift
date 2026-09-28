import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        NavigationSplitView {
            ControlPanel()
                .navigationSplitViewColumnWidth(min: 360, ideal: 430, max: 600)
        } detail: {
            OutputPanel()
        }
        .navigationTitle("Turbo MLX")
        // The name stays the window's (Window menu, Mission Control) but not in the toolbar.
        .toolbar(removing: .title)
        .alert(
            app.alert?.title ?? "",
            isPresented: Binding(get: { app.alert != nil }, set: { if !$0 { app.alert = nil } }),
            presenting: app.alert
        ) { alert in
            if let link = alert.link {
                Button(link.title) { NSWorkspace.shared.open(link.url) }
            }
            if alert.offersLog {
                Button("Open Log") { openWindow(id: WindowID.log) }
            }
            Button(alert.link == nil ? "OK" : "Later", role: .cancel) {}
        } message: { alert in
            Text(alert.message)
        }
    }
}
