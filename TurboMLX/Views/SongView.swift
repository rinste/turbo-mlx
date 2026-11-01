import AppKit
import AVFoundation
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// A song's waveform: how loud each stretch of it is (RMS, both channels together: peaks would sit at the top all
/// through a loud song), read from its WAV a block at a time (six minutes of 44.1 kHz stereo would be 127 MB of
/// floats at once).
nonisolated enum SongWaveform {
    struct ReadError: LocalizedError {
        var errorDescription: String? { "The song’s file could not be read." }
    }

    /// `buckets` levels in 0…1, and the song's length in seconds.
    static func levels(of url: URL, buckets: Int) throws -> (levels: [Float], seconds: Double) {
        let file = try AVAudioFile(forReading: url)
        let frames = Int(file.length)
        guard frames > 0, buckets > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1 << 16)
        else { throw ReadError() }
        var squares = [Double](repeating: 0, count: buckets)
        var counts = [Int](repeating: 0, count: buckets)
        var read = 0
        while read < frames {
            try file.read(into: buffer)
            let count = Int(buffer.frameLength)
            guard count > 0, let channels = buffer.floatChannelData else { break }
            for channel in 0 ..< Int(buffer.format.channelCount) {
                let samples = channels[channel]
                for index in 0 ..< count {
                    let bucket = min((read + index) * buckets / frames, buckets - 1)
                    squares[bucket] += Double(samples[index] * samples[index])
                    counts[bucket] += 1
                }
            }
            read += count
        }
        let levels = zip(squares, counts).map { Float(($0 / Double(max($1, 1))).squareRoot()) }
        return (levels, Double(frames) / file.processingFormat.sampleRate)
    }

    /// The picture the history shows for a song: its waveform, light bars on a deep gradient, as a square PNG.
    static func write(of wav: URL, to url: URL, size: Int = 768) throws {
        let bars = 64
        let (levels, _) = try Self.levels(of: wav, buckets: bars)
        let loudest = max(levels.max() ?? 0, 1e-4)
        guard let context = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ReadError() }
        let side = CGFloat(size)
        let colors = [CGColor(srgbRed: 0.33, green: 0.18, blue: 0.62, alpha: 1), CGColor(srgbRed: 0.10, green: 0.28, blue: 0.55, alpha: 1)]
        if let gradient = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colors as CFArray, locations: [0, 1]) {
            context.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: side, y: side), options: [])
        }
        let margin = side * 0.08, width = (side - 2 * margin) / CGFloat(bars)
        context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.9))
        for (index, level) in levels.enumerated() {
            let height = max(side * 0.012, CGFloat(level / loudest) * side * 0.6)
            let bar = CGRect(x: margin + CGFloat(index) * width + width * 0.18, y: (side - height) / 2, width: width * 0.64, height: height)
            context.addPath(CGPath(roundedRect: bar, cornerWidth: width * 0.32, cornerHeight: min(width * 0.32, height / 2), transform: nil))
        }
        context.fillPath()
        guard let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ReadError() }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ReadError() }
    }
}

/// The song being listened to: an `AVAudioPlayer`, and where it is.
@Observable
final class SongPlayer: NSObject, AVAudioPlayerDelegate {
    private(set) var isPlaying = false
    private var player: AVAudioPlayer?
    /// Where playback starts, or stopped.
    private var position = 0.0

    var isReady: Bool { player != nil }

    func load(_ url: URL) {
        stop()
        player = try? AVAudioPlayer(contentsOf: url)
        player?.delegate = self
        player?.prepareToPlay()
    }

    /// Seconds into the song.
    var currentTime: Double { isPlaying ? player?.currentTime ?? position : position }

    func toggle() {
        guard let player else { return }
        if isPlaying {
            player.pause()
            position = player.currentTime
            isPlaying = false
        } else {
            player.currentTime = position
            isPlaying = player.play()
        }
    }

    func seek(to seconds: Double) {
        position = max(seconds, 0)
        if isPlaying { player?.currentTime = position }
    }

    func stop() {
        player?.stop()
        isPlaying = false
        position = 0
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        position = 0
    }
}

/// A song in the output panel: its waveform, the part played so far in the accent colour, a click or a drag on it to
/// move there, and a button to play or pause (Space). Nothing plays until asked.
struct SongStage: View {
    let player: SongPlayer
    let url: URL
    /// The panel takes the keyboard back after a click, so that Space keeps reaching the song.
    var onInteraction: () -> Void = {}

    @State private var levels: [Float] = []
    @State private var seconds = 0.0
    @State private var waveformWidth: CGFloat = 1

    var body: some View {
        VStack(spacing: 18) {
            TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !player.isPlaying)) { _ in
                waveform(played: seconds > 0 ? min(player.currentTime / seconds, 1) : 0)
            }
            .frame(maxWidth: 900, minHeight: 120, maxHeight: 220)
            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { waveformWidth = max($0, 1) }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                player.seek(to: Double(min(max(value.location.x / waveformWidth, 0), 1)) * seconds)
                onInteraction()
            })
            .accessibilityLabel("Waveform")
            .accessibilityValue("\(Format.songDuration(player.currentTime)) of \(Format.songDuration(seconds))")

            HStack(spacing: 14) {
                Button {
                    player.toggle()
                    onInteraction()
                } label: {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44))
                        .symbolRenderingMode(.hierarchical)
                }
                .buttonStyle(.plain)
                .help(player.isPlaying ? "Pause (Space)" : "Play (Space)")
                .accessibilityLabel(player.isPlaying ? "Pause" : "Play")
                .disabled(!player.isReady)

                TimelineView(.periodic(from: .now, by: 0.25)) { _ in
                    Text(verbatim: "\(Format.songDuration(player.currentTime)) / \(Format.songDuration(seconds))")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: url) { await load() }
        .onDisappear { player.stop() }
    }

    private func waveform(played: Double) -> some View {
        Canvas { context, size in
            guard !levels.isEmpty else { return }
            let loudest = max(levels.max() ?? 0, 1e-4)
            let width = size.width / CGFloat(levels.count)
            for (index, level) in levels.enumerated() {
                let height = max(2, CGFloat(level / loudest) * size.height)
                let bar = CGRect(x: CGFloat(index) * width + width * 0.15, y: (size.height - height) / 2, width: max(width * 0.7, 1), height: height)
                let isPlayed = (CGFloat(index) + 0.5) / CGFloat(levels.count) <= played
                context.fill(Path(roundedRect: bar, cornerRadius: min(width * 0.35, 2)),
                             with: .color(isPlayed ? Color.accentColor : Color.secondary.opacity(0.45)))
            }
        }
    }

    private func load() async {
        let url = url
        let read = await Task.detached { try? SongWaveform.levels(of: url, buckets: 240) }.value
        levels = read?.levels ?? []
        seconds = read?.seconds ?? 0
        player.load(url)
    }
}
