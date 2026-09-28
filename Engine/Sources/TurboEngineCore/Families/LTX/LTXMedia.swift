import Accelerate
import CoreGraphics
import Foundation
import ImageIO
import MLX

// The pixels around the LTX pipeline: the reference image that pins the first frame, and the
// video decoder run in tiles when a whole clip would not fit in memory.

/// A reference image: read once, then resized and cropped to each stage's size and encoded
/// (`load_image_and_preprocess` + `VideoEncoder.encode`, as `combined_image_conditionings`
/// builds the first frame's tokens).
struct LTXReferenceImage {
    let image: CGImage

    init(path: String) throws {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { throw LTXError.unreadableImage(path) }
        self.image = image
    }

    /// The first frame's latent tokens [1, h·w, 128] at `width` × `height` pixels.
    func tokens(width: Int, height: Int, encoder: LTXVideoEncoder) -> MLXArray {
        let rgb = Self.coverAndCrop(image, width: width, height: height)
        let pixels = (rgb.asType(.float32) / Float(255)) * Float(2) - Float(1)
        let tensor = pixels.transposed(2, 0, 1).reshaped([1, 3, 1, height, width]).asType(.bfloat16)
        let latent = encoder.encode(tensor)
        eval(latent)
        return latent.transposed(0, 2, 3, 4, 1).reshaped([1, -1, 128])
    }

    /// `resize_and_center_crop`: scaled to cover `width` × `height` (a Lanczos-class filter), the
    /// middle cut out. [H, W, 3] uint8.
    static func coverAndCrop(_ image: CGImage, width: Int, height: Int) -> MLXArray {
        let scale = max(Double(height) / Double(image.height), Double(width) / Double(image.width))
        let scaledWidth = Int((Double(image.width) * scale).rounded(.up))
        let scaledHeight = Int((Double(image.height) * scale).rounded(.up))
        let rgba = rgbaBytes(image)
        let resized = scaledWidth == image.width && scaledHeight == image.height
            ? rgba
            : resize(rgba, width: image.width, height: image.height, toWidth: scaledWidth, toHeight: scaledHeight)
        let left = (scaledWidth - width) / 2
        let top = (scaledHeight - height) / 2
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let source = ((top + y) * scaledWidth + (left + x)) * 4
                let target = (y * width + x) * 3
                rgb[target] = resized[source]
                rgb[target + 1] = resized[source + 1]
                rgb[target + 2] = resized[source + 2]
            }
        }
        return MLXArray(rgb, [height, width, 3])
    }

    /// The image as RGBA bytes on an sRGB canvas, transparency over white.
    private static func rgbaBytes(_ image: CGImage) -> [UInt8] {
        let (width, height) = (image.width, image.height)
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        bytes.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return bytes
    }

    private static func resize(_ bytes: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        var input = bytes
        var output = [UInt8](repeating: 0, count: toWidth * toHeight * 4)
        input.withUnsafeMutableBytes { inBuffer in
            output.withUnsafeMutableBytes { outBuffer in
                var source = vImage_Buffer(data: inBuffer.baseAddress, height: vImagePixelCount(height),
                                           width: vImagePixelCount(width), rowBytes: width * 4)
                var destination = vImage_Buffer(data: outBuffer.baseAddress, height: vImagePixelCount(toHeight),
                                                width: vImagePixelCount(toWidth), rowBytes: toWidth * 4)
                _ = vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        return output
    }
}

/// The video decode in tiles (`_compute_decode_tiling`, `prepare_tiles_for_decoding`,
/// `VideoDecoder.tiled_decode`): temporal tiles of latent frames and, if needed, spatial tiles,
/// each decoded alone and blended with trapezoidal ramps; the frames leave in temporal chunks.
enum LTXDecodeTiling {
    struct Config: Equatable {
        /// Pixels per tile side and their overlap.
        var spatial: (tile: Int, overlap: Int)?
        /// Frames per tile and their overlap.
        var temporal: (tile: Int, overlap: Int)?

        static func == (a: Config, b: Config) -> Bool {
            a.spatial?.tile == b.spatial?.tile && a.spatial?.overlap == b.spatial?.overlap
                && a.temporal?.tile == b.temporal?.tile && a.temporal?.overlap == b.temporal?.overlap
        }
    }

    /// Peak activation bytes of the decoder per output pixel-frame (measured by the reference).
    static let bytesPerPixelFrame = 750
    static let accumulatorBytesPerPixel = 4 * 3 * 4
    static let spatialLadder = [(768, 64), (512, 32), (256, 32)]

    /// The reference's budget, half of the unified memory, kept within what the models already
    /// resident leave of three quarters of it.
    static func budget(resident: Int) -> Int {
        let memory = Int(ProcessInfo.processInfo.physicalMemory)
        return max(min(memory / 2, memory * 3 / 4 - resident), 3 << 30)
    }

    private static func spatialTilePixels(axis: Int, longSide: Int, tile: Int, overlap: Int) -> Int {
        let tileLatent = tile / 32
        let overlapLatent = overlap / 32
        let adjusted = max(max(2, overlapLatent + 1), roundHalfEven(Double(tileLatent * axis) / Double(longSide)))
        return min(axis, adjusted) * 32
    }

    static func estimate(latentShape shape: [Int], config: Config?) -> Int {
        let (f, h, w) = (shape[2], shape[3], shape[4])
        let (fp, hp, wp) = (8 * f - 7, 32 * h, 32 * w)
        guard let config else { return bytesPerPixelFrame * fp * hp * wp }
        var (tf, th, tw) = (fp, hp, wp)
        if let temporal = config.temporal { tf = min(fp, temporal.tile) }
        if let spatial = config.spatial {
            let longSide = max(h, w)
            th = spatialTilePixels(axis: h, longSide: longSide, tile: spatial.tile, overlap: spatial.overlap)
            tw = spatialTilePixels(axis: w, longSide: longSide, tile: spatial.tile, overlap: spatial.overlap)
        }
        return bytesPerPixelFrame * tf * th * tw + tf * hp * wp * accumulatorBytesPerPixel
    }

    /// A temporal tile of `frames` blended over about a second, at most 30% of the tile.
    private static func temporalTile(_ frames: Int, fps: Double) -> (tile: Int, overlap: Int) {
        let second = max(8, (Int(fps) / 8) * 8)
        return (frames, min(second, (Int(Double(frames) * 0.3) / 8) * 8))
    }

    /// No tiling when the whole clip fits the budget; otherwise the first rung that fits: temporal
    /// tiles of 80 down to 40 frames, then 768 / 512 / 256-pixel tiles at 40 frames, then shorter
    /// temporal tiles at 256 pixels.
    static func plan(latentShape: [Int], fps: Double, budget: Int) -> Config? {
        guard estimate(latentShape: latentShape, config: nil) > budget else { return nil }
        let fp = 8 * latentShape[2] - 7
        let sizes = Swift.stride(from: 80, through: 16, by: -8).filter { $0 < fp }
        let preferred = sizes.filter { $0 >= 40 }
        let smallest = sizes.filter { $0 < 40 }
        var candidates = preferred.map { Config(spatial: nil, temporal: temporalTile($0, fps: fps)) }
        let base = preferred.last.map { temporalTile($0, fps: fps) }
        for (pixels, overlap) in spatialLadder { candidates.append(Config(spatial: (pixels, overlap), temporal: base)) }
        let (lastPixels, lastOverlap) = spatialLadder[spatialLadder.count - 1]
        for frames in smallest { candidates.append(Config(spatial: (lastPixels, lastOverlap), temporal: temporalTile(frames, fps: fps))) }
        return candidates.first { estimate(latentShape: latentShape, config: $0) <= budget } ?? candidates.last
    }

    // MARK: Tiles

    struct Interval {
        var starts: [Int]
        var ends: [Int]
        var leftRamps: [Int]
        var rightRamps: [Int]
    }

    struct Tile {
        /// Latent ranges: frames, rows, columns.
        var input: (Range<Int>, Range<Int>, Range<Int>)
        /// Output ranges in pixels (nil: the whole axis).
        var frames: Range<Int>?
        var rows: Range<Int>?
        var columns: Range<Int>?
        var frameMask: [Float]?
        var rowMask: [Float]?
        var columnMask: [Float]?
    }

    static func symmetricSplit(_ length: Int, size: Int, overlap: Int) -> Interval {
        guard length > size else { return Interval(starts: [0], ends: [length], leftRamps: [0], rightRamps: [0]) }
        let amount = (length + size - 2 * overlap - 1) / (size - overlap)
        let starts = (0 ..< amount).map { $0 * (size - overlap) }
        var ends = starts.map { $0 + size }
        ends[amount - 1] = length
        return Interval(starts: starts, ends: ends, leftRamps: [0] + Array(repeating: overlap, count: amount - 1),
                        rightRamps: Array(repeating: overlap, count: amount - 1) + [0])
    }

    /// `split_temporal_latents`: later tiles start one latent frame early, their ramp one longer.
    static func temporalSplit(_ length: Int, size: Int, overlap: Int) -> Interval {
        guard length > size else { return Interval(starts: [0], ends: [length], leftRamps: [0], rightRamps: [0]) }
        var interval = symmetricSplit(length, size: size, overlap: overlap)
        for i in 1 ..< interval.starts.count {
            interval.starts[i] -= 1
            interval.leftRamps[i] += 1
        }
        return interval
    }

    /// `compute_trapezoidal_mask_1d`.
    static func trapezoid(length: Int, left: Int, right: Int, leftFromZero: Bool) -> [Float] {
        let left = max(0, min(left, length))
        let right = max(0, min(right, length))
        var mask = [Float](repeating: 1, count: length)
        if left > 0 {
            let count = leftFromZero ? left + 1 : left + 2
            var fade = Array(linspace(0, 1, count).dropLast())
            if !leftFromZero { fade = Array(fade.dropFirst()) }
            for i in 0 ..< left { mask[i] *= fade[i] }
        }
        if right > 0 {
            let fade = Array(linspace(1, 0, right + 2).dropFirst().dropLast())
            for i in 0 ..< right { mask[length - right + i] *= fade[i] }
        }
        return mask.map { min(max($0, 0), 1) }
    }

    /// MLX's float32 linspace: (1 − t)·start + t·stop with t = i / (n − 1).
    private static func linspace(_ start: Float, _ stop: Float, _ count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        return (0 ..< count).map { i in
            let t = Float(i) / Float(count - 1)
            return (1 - t) * start + t * stop
        }
    }

    static func tiles(latentShape shape: [Int], config: Config?) -> [Tile] {
        let (f, h, w) = (shape[2], shape[3], shape[4])
        var frameIntervals = Interval(starts: [0], ends: [f], leftRamps: [0], rightRamps: [0])
        var rowIntervals = Interval(starts: [0], ends: [h], leftRamps: [0], rightRamps: [0])
        var columnIntervals = Interval(starts: [0], ends: [w], leftRamps: [0], rightRamps: [0])
        var temporalTiled = false
        var spatialTiled = false
        if let spatial = config?.spatial {
            let longSide = max(h, w)
            let tile = spatial.tile / 32
            let overlap = spatial.overlap / 32
            let lower = max(2, overlap + 1)
            rowIntervals = symmetricSplit(h, size: max(lower, roundHalfEven(Double(tile * h) / Double(longSide))), overlap: overlap)
            columnIntervals = symmetricSplit(w, size: max(lower, roundHalfEven(Double(tile * w) / Double(longSide))), overlap: overlap)
            spatialTiled = true
        }
        if let temporal = config?.temporal {
            frameIntervals = temporalSplit(f, size: temporal.tile / 8, overlap: temporal.overlap / 8)
            temporalTiled = true
        }

        var tiles: [Tile] = []
        for fi in frameIntervals.starts.indices {
            for ri in rowIntervals.starts.indices {
                for ci in columnIntervals.starts.indices {
                    var tile = Tile(input: (frameIntervals.starts[fi] ..< frameIntervals.ends[fi],
                                            rowIntervals.starts[ri] ..< rowIntervals.ends[ri],
                                            columnIntervals.starts[ci] ..< columnIntervals.ends[ci]))
                    if temporalTiled {
                        let (begin, end) = (frameIntervals.starts[fi], frameIntervals.ends[fi])
                        let (left, right) = (frameIntervals.leftRamps[fi], frameIntervals.rightRamps[fi])
                        let start = begin * 8
                        let stop = 1 + (end - 1) * 8
                        tile.frames = start ..< stop
                        tile.frameMask = trapezoid(length: stop - start, left: left == 0 ? 0 : 1 + (left - 1) * 8,
                                                   right: right * 8, leftFromZero: true)
                    }
                    if spatialTiled {
                        let rows = (rowIntervals.starts[ri] * 32) ..< (rowIntervals.ends[ri] * 32)
                        tile.rows = rows
                        tile.rowMask = trapezoid(length: rows.count, left: rowIntervals.leftRamps[ri] * 32,
                                                 right: rowIntervals.rightRamps[ri] * 32, leftFromZero: false)
                        let columns = (columnIntervals.starts[ci] * 32) ..< (columnIntervals.ends[ci] * 32)
                        tile.columns = columns
                        tile.columnMask = trapezoid(length: columns.count, left: columnIntervals.leftRamps[ci] * 32,
                                                    right: columnIntervals.rightRamps[ci] * 32, leftFromZero: false)
                    }
                    tiles.append(tile)
                }
            }
        }
        return tiles
    }

    // MARK: Decoding

    /// Decodes `latent` [1, 128, F, H, W] and hands `emit` the pixels [1, 3, N, H, W] in [-1, 1],
    /// frame chunks in order.
    static func decode(
        _ latent: MLXArray, decoder: LTXVideoDecoder, tiling: Config?, isCancelled: () -> Bool,
        emit: (MLXArray) throws -> Void
    ) throws {
        guard let tiling else {
            let pixels = decoder.decode(latent)
            eval(pixels)
            try emit(pixels)
            return
        }
        let shape = latent.shape
        let (f, h, w) = (shape[2], shape[3], shape[4])
        let (outHeight, outWidth) = (h * 32, w * 32)
        let framesOut = 8 * f - 7
        let tiles = tiles(latentShape: shape, config: tiling)

        // Tiles grouped by their output frames, in order.
        var groups: [[Tile]] = []
        for tile in tiles {
            if let last = groups.last?.first, last.frames == tile.frames {
                groups[groups.count - 1].append(tile)
            } else {
                groups.append([tile])
            }
        }

        var previous: (chunk: MLXArray, weights: MLXArray, frames: Range<Int>)?
        for group in groups {
            let current = group[0].frames ?? 0 ..< framesOut
            let length = current.count
            var buffer = MLXArray.zeros([1, 3, length, outHeight, outWidth])
            var weights = MLXArray.zeros([1, 3, length, outHeight, outWidth])
            for tile in group {
                if isCancelled() { throw GenerationError.cancelled }
                let (fr, rr, cr) = tile.input
                let decoded = decoder.decode(latent[0..., 0..., fr, rr, cr], evaluateStages: true)
                eval(decoded)
                let tileFrames = tile.frames ?? 0 ..< framesOut
                let offset = tileFrames.lowerBound - current.lowerBound
                let actual = min(tileFrames.count, decoded.shape[2], length - offset)
                let rows = tile.rows ?? 0 ..< outHeight
                let columns = tile.columns ?? 0 ..< outWidth
                var mask = MLXArray(Float(1)).reshaped([1, 1, 1, 1, 1])
                if let frameMask = tile.frameMask { mask = mask * MLXArray(Array(frameMask.prefix(actual))).reshaped([1, 1, actual, 1, 1]) }
                if let rowMask = tile.rowMask { mask = mask * MLXArray(rowMask).reshaped([1, 1, 1, rowMask.count, 1]) }
                if let columnMask = tile.columnMask { mask = mask * MLXArray(columnMask).reshaped([1, 1, 1, 1, columnMask.count]) }
                let slice = decoded[0..., 0..., 0 ..< actual]
                let frames = offset ..< (offset + actual)
                buffer[0..., 0..., frames, rows, columns] = buffer[0..., 0..., frames, rows, columns] + slice * mask
                weights[0..., 0..., frames, rows, columns] = weights[0..., 0..., frames, rows, columns] + mask
                eval(buffer, weights)
            }

            if var previousGroup = previous {
                if previousGroup.frames.upperBound > current.lowerBound {
                    let overlap = previousGroup.frames.upperBound - current.lowerBound
                    let previousStart = current.lowerBound - previousGroup.frames.lowerBound
                    let merged = previousGroup.chunk[0..., 0..., previousStart...] + buffer[0..., 0..., 0 ..< overlap]
                    let mergedWeights = previousGroup.weights[0..., 0..., previousStart...] + weights[0..., 0..., 0 ..< overlap]
                    previousGroup.chunk = concatenated([previousGroup.chunk[0..., 0..., 0 ..< previousStart], merged], axis: 2)
                    previousGroup.weights = concatenated([previousGroup.weights[0..., 0..., 0 ..< previousStart], mergedWeights], axis: 2)
                    buffer = concatenated([merged, buffer[0..., 0..., overlap...]], axis: 2)
                    weights = concatenated([mergedWeights, weights[0..., 0..., overlap...]], axis: 2)
                }
                let emitted = current.lowerBound - previousGroup.frames.lowerBound
                if emitted > 0 {
                    let chunk = (previousGroup.chunk / maximum(previousGroup.weights, MLXArray(Float(1e-8))))[0..., 0..., 0 ..< emitted]
                    eval(chunk)
                    try emit(chunk)
                }
            }
            previous = (buffer, weights, current)
        }
        if let previous {
            let chunk = previous.chunk / maximum(previous.weights, MLXArray(Float(1e-8)))
            eval(chunk)
            try emit(chunk)
        }
    }
}

/// Python's `round()`: halves to even.
func roundHalfEven(_ value: Double) -> Int {
    Int(value.rounded(.toNearestOrEven))
}
