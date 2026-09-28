import AppKit
import SwiftUI

/// The classic transparency pattern, drawn behind RGBA images.
struct Checkerboard: View {
    var squareSize: CGFloat = 10
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Canvas { context, size in
            let light = colorScheme == .dark ? Color(white: 0.24) : Color(white: 1)
            let dark = colorScheme == .dark ? Color(white: 0.18) : Color(white: 0.88)
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(light))
            var squares = Path()
            let columns = Int((size.width / squareSize).rounded(.up))
            let rows = Int((size.height / squareSize).rounded(.up))
            for row in 0..<rows {
                for column in 0..<columns where (row + column).isMultiple(of: 2) {
                    squares.addRect(CGRect(
                        x: CGFloat(column) * squareSize,
                        y: CGFloat(row) * squareSize,
                        width: squareSize,
                        height: squareSize
                    ))
                }
            }
            context.fill(squares, with: .color(dark))
        }
    }
}

/// An image file decoded off the main thread; `maxPixelSize` nil shows it at full resolution.
struct FileImage: View {
    let url: URL
    var maxPixelSize: Int?

    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
            } else {
                Color.clear
            }
        }
        .task(id: url) {
            if let cached = ImageLoader.cached(url, maxPixelSize: maxPixelSize) {
                image = cached
            } else {
                image = await ImageLoader.load(url, maxPixelSize: maxPixelSize)
            }
        }
    }
}

/// A small label + value pill used for image metadata.
struct MetadataChip: View {
    let systemImage: String
    let text: String
    var help: String?

    var body: some View {
        Label(text, systemImage: systemImage)
            .labelStyle(.titleAndIcon)
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(.quaternary.opacity(0.6), in: Capsule())
            .help(help ?? text)
    }
}

extension Color {
    /// What is active or selected: a soft white in the dark appearance, a dark gray in the light
    /// one, next to the gray accent of the other controls (Assets → ActiveColor).
    static let active = Color("ActiveColor")
}

/// The main action of a view, drawn in the active color with its text in the opposite tone;
/// disabled, it drops back to a quiet gray.
struct PrimaryActionButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    @Environment(\.colorScheme) private var colorScheme

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(isEnabled ? labelColor : Color.secondary)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .frame(minHeight: 36)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isEnabled ? Color.active : Color.primary.opacity(0.08))
            )
            .opacity(configuration.isPressed ? 0.75 : 1)
            .contentShape(RoundedRectangle(cornerRadius: 8))
    }

    /// Dark text on the soft white, white text on the dark gray.
    private var labelColor: Color {
        colorScheme == .dark ? Color(white: 0.1) : Color(white: 0.98)
    }
}

extension ButtonStyle where Self == PrimaryActionButtonStyle {
    static var primaryAction: PrimaryActionButtonStyle { PrimaryActionButtonStyle() }
}

/// A rectangle drawn in a given aspect ratio, used by the format picker.
struct AspectGlyph: View {
    let ratio: Double
    var isSelected = false

    var body: some View {
        let side: CGFloat = 22
        let width = ratio >= 1 ? side : side * ratio
        let height = ratio >= 1 ? side / ratio : side
        RoundedRectangle(cornerRadius: 3)
            .strokeBorder(isSelected ? Color.active : Color.secondary, lineWidth: 1.5)
            .background(
                RoundedRectangle(cornerRadius: 3)
                    .fill(isSelected ? Color.active.opacity(0.18) : .clear)
            )
            .frame(width: width, height: height)
            .frame(width: side, height: side)
    }
}

enum Format {
    static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }

    static func memory(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .memory)
    }

    /// "42 s", "3 min 05 s", "1 h 02 min".
    static func duration(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total < 60 { return "\(total) s" }
        if total < 3600 { return String(format: "%d min %02d s", total / 60, total % 60) }
        return String(format: "%d h %02d min", total / 3600, (total % 3600) / 60)
    }

    /// Like `duration`, with a decimal under ten seconds: "0.4 s", "7.5 s", "42 s".
    static func seconds(_ value: Double) -> String {
        value < 10 ? "\(value.formatted(.number.precision(.fractionLength(1)))) s" : duration(value)
    }

    /// "~45 s", "~4 min", "~1 h 10 min": a duration no more precise than a guess can be.
    static func estimate(_ seconds: Double) -> String {
        if seconds < 100 { return "~\(max(5, Int((seconds / 5).rounded()) * 5)) s" }
        if seconds < 3570 { return "~\(Int((seconds / 60).rounded())) min" }
        let minutes = Int((seconds / 300).rounded()) * 5
        return minutes % 60 == 0 ? "~\(minutes / 60) h" : "~\(minutes / 60) h \(minutes % 60) min"
    }

    static func remaining(_ seconds: Double) -> String {
        seconds < 5 ? "almost done" : "about \(duration(seconds)) left"
    }

    static func guidance(_ value: Double) -> String {
        value <= 1 ? "Off" : value.formatted(.number.precision(.fractionLength(1)))
    }

    /// "5 s", "2.5 s": a clip's length from its frames.
    static func clipDuration(frames: Int, fps: Int) -> String {
        let seconds = Double(frames - 1) / Double(fps)
        let rounded = (seconds * 10).rounded() / 10
        return rounded == rounded.rounded()
            ? "\(Int(rounded)) s"
            : "\(rounded.formatted(.number.precision(.fractionLength(1)))) s"
    }
}

extension NSPasteboard {
    /// An image as `copyImage` copies it; a clip as its file.
    func copyItem(_ item: HistoryItem, at url: URL) {
        guard item.kind == .video else { return copyImage(at: url) }
        clearContents()
        writeObjects([url as NSURL])
    }

    /// Copies a PNG so that transparency survives, plus the file itself for Finder.
    func copyImage(at url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        clearContents()
        declareTypes([.png, .tiff, .fileURL], owner: nil)
        setData(data, forType: .png)
        if let tiff = NSImage(data: data)?.tiffRepresentation { setData(tiff, forType: .tiff) }
        setString(url.absoluteString, forType: .fileURL)
    }
}
