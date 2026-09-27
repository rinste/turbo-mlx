import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        NavigationSplitView {
            ControlPanel()
                .navigationSplitViewColumnWidth(min: 330, ideal: 370, max: 480)
        } detail: {
            OutputPanel()
        }
        .navigationTitle("Turbo MLX")
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
    }
}
