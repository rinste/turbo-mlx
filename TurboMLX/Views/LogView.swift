import SwiftUI

/// Everything the Python side printed: mflux output, warnings and tracebacks.
struct LogView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        let log = app.backend.log
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(log.lines) { line in
                        Text(verbatim: line.text)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(Self.isError(line.text) ? .red : .primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(line.id)
                    }
                }
                .padding(10)
                .textSelection(.enabled)
            }
            .onChange(of: log.lines.last?.id) { _, id in
                if let id { proxy.scrollTo(id, anchor: .bottom) }
            }
            .onAppear {
                if let id = log.lines.last?.id { proxy.scrollTo(id, anchor: .bottom) }
            }
        }
        .overlay {
            if log.lines.isEmpty {
                ContentUnavailableView("The Log Is Empty", systemImage: "text.alignleft",
                                       description: Text("Output from mflux and the Python engine appears here."))
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(log.text, forType: .string)
                } label: {
                    Label("Copy All", systemImage: "doc.on.doc")
                }
                Button {
                    log.clear()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
            }
        }
    }

    private static func isError(_ text: String) -> Bool {
        text.hasPrefix("Traceback") || text.contains("Error:") || text.hasPrefix("error")
    }
}
