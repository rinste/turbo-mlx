import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

/// `turbo-engine compare a b`: how far two results are apart, for a real image or clip made by
/// the engine and by a reference (or by two builds of the engine). Images: the PSNR over their
/// channels and the largest difference, from the decoded file's own values (no color management).
/// Clips: the same frame by frame, and the sound's RMS difference against its level.
enum Compare {
    static func run(_ first: URL, _ second: URL) -> Bool {
        let video = ["mp4", "mov", "m4v"].contains(first.pathExtension.lowercased())
        do {
            let line = video ? try clips(first, second) : try images(first, second)
            print(line)
            return true
        } catch {
            print("compare failed: \(error.localizedDescription)")
            return false
        }
    }

    struct CompareError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    // MARK: Images

    private static func images(_ first: URL, _ second: URL) throws -> String {
        let (a, width, height, channels) = try pixels(first)
        let (b, width2, height2, channels2) = try pixels(second)
        guard width == width2, height == height2, channels == channels2 else {
            throw CompareError("the images differ in size: \(width) × \(height) × \(channels) and \(width2) × \(height2) × \(channels2)")
        }
        let (psnr, largest) = difference(a, b)
        return "\(format(psnr)), largest difference \(largest) (\(width) × \(height) × \(channels))"
    }

    /// The image's 8-bit channels as stored: RGB or RGBA, row after row.
    private static func pixels(_ url: URL) throws -> ([UInt8], Int, Int, Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.bitsPerComponent == 8, let data = image.dataProvider?.data as Data?
        else { throw CompareError("cannot read \(url.path) as an 8-bit image") }
        let stride = image.bitsPerPixel / 8
        let alpha = image.alphaInfo
        let hasAlpha = [.first, .last, .premultipliedFirst, .premultipliedLast].contains(alpha)
        // Where red starts in each pixel, and where alpha is, for the layouts PNGs decode to.
        let colorOffset = [.first, .premultipliedFirst, .noneSkipFirst].contains(alpha) ? 1 : 0
        let alphaOffset = [.first, .premultipliedFirst].contains(alpha) ? 0 : 3
        let channels = hasAlpha ? 4 : 3
        var out = [UInt8]()
        out.reserveCapacity(image.width * image.height * channels)
        data.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for y in 0 ..< image.height {
                for x in 0 ..< image.width {
                    let pixel = y * image.bytesPerRow + x * stride
                    if stride == 1 {
                        out += [bytes[pixel], bytes[pixel], bytes[pixel]]
                        continue
                    }
                    out += [bytes[pixel + colorOffset], bytes[pixel + colorOffset + 1], bytes[pixel + colorOffset + 2]]
                    if hasAlpha { out.append(bytes[pixel + alphaOffset]) }
                }
            }
        }
        return (out, image.width, image.height, stride == 1 ? 3 : channels)
    }

    // MARK: Clips

    private static func clips(_ first: URL, _ second: URL) throws -> String {
        let a = try frames(first)
        let b = try frames(second)
        guard a.width == b.width, a.height == b.height else { throw CompareError("the clips differ in size") }
        let count = min(a.frames.count, b.frames.count)
        guard count > 0 else { throw CompareError("no frames to compare") }
        var total = 0.0
        var largest = 0
        var identical = 0
        for index in 0 ..< count {
            let (psnr, frameLargest) = difference(a.frames[index], b.frames[index])
            if psnr.isInfinite { identical += 1 } else { total += psnr }
            largest = max(largest, frameLargest)
        }
        let differing = count - identical
        let framesLine = differing == 0
            ? "\(count) frames, identical"
            : "\(count) frames, mean PSNR \(String(format: "%.1f", total / Double(differing))) dB over the \(differing) that differ, largest difference \(largest)"
        var line = "\(framesLine) (\(a.width) × \(a.height); \(a.frames.count) and \(b.frames.count) frames)"
        let soundA = try sound(first)
        let soundB = try sound(second)
        if !soundA.isEmpty, !soundB.isEmpty {
            let n = min(soundA.count, soundB.count)
            var difference = 0.0
            var level = 0.0
            for index in 0 ..< n {
                let d = Double(soundA[index] - soundB[index])
                difference += d * d
                level += Double(soundA[index] * soundA[index])
            }
            let relative = level > 0 ? (difference / level).squareRoot() : 0
            line += "; sound RMS difference \(String(format: "%.3f", relative)) of its level"
        }
        return line
    }

    /// Every frame of the clip's video track as RGB bytes.
    private static func frames(_ url: URL) throws -> (frames: [[UInt8]], width: Int, height: Int) {
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else { throw CompareError("\(url.path) has no video track") }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(output)
        reader.startReading()
        var frames: [[UInt8]] = []
        var size = (0, 0)
        while let sample = output.copyNextSampleBuffer(), let buffer = CMSampleBufferGetImageBuffer(sample) {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            let width = CVPixelBufferGetWidth(buffer)
            let height = CVPixelBufferGetHeight(buffer)
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            var rgb = [UInt8]()
            rgb.reserveCapacity(width * height * 3)
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let pixel = base + y * rowBytes + x * 4
                    rgb += [pixel[2], pixel[1], pixel[0]]
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            frames.append(rgb)
            size = (width, height)
        }
        return (frames, size.0, size.1)
    }

    /// The clip's sound as interleaved float samples, empty without a sound track.
    private static func sound(_ url: URL) throws -> [Float] {
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .audio).first else { return [] }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
        ])
        reader.add(output)
        reader.startReading()
        var samples: [Float] = []
        while let sample = output.copyNextSampleBuffer(), let block = CMSampleBufferGetDataBuffer(sample) {
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / 4)
            _ = chunk.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            samples += chunk
        }
        return samples
    }

    // MARK: Numbers

    private static func difference(_ a: [UInt8], _ b: [UInt8]) -> (psnr: Double, largest: Int) {
        var squares = 0.0
        var largest = 0
        for index in 0 ..< min(a.count, b.count) {
            let d = Int(a[index]) - Int(b[index])
            squares += Double(d * d)
            largest = max(largest, abs(d))
        }
        let mse = squares / Double(max(min(a.count, b.count), 1))
        return (mse == 0 ? .infinity : 10 * log10(255 * 255 / mse), largest)
    }

    private static func format(_ psnr: Double) -> String {
        psnr.isInfinite ? "identical" : String(format: "PSNR %.1f dB", psnr)
    }
}
