import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        NavigationSplitView {
            ControlPanel()
                .navigationSplitViewColumnWidth(min: 360, ideal: 470, max: 600)
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
            if alert.offersLog {
                Button("Open Log") { openWindow(id: WindowID.log) }
            }
            Button("OK", role: .cancel) {}
        } message: { alert in
            Text(alert.message)
        }
        // Describe Image's first picture: the describer is a download of its own.
        .confirmationDialog(
            "Download \(ModelCatalog.describer.name) to describe pictures?",
            isPresented: Binding(get: { app.describerOffer != nil }, set: { if !$0 { app.describerOffer = nil } }),
            presenting: app.describerOffer
        ) { picture in
            Button("Download and Describe") { app.downloadDescriber(then: picture) }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("Describing pictures uses \(ModelCatalog.describer.name), a \(Format.bytes(ModelCatalog.describerBytes)) download from Hugging Face (\(ModelCatalog.describer.license) license). It is downloaded once and stays on this Mac; Settings → Models can move it to the Trash.")
        }
    }
}
