import Foundation
import ImageIO
import MLX
import MLXNN

/// The picture an upscale starts from, as `SeedVR2Util.preprocess_image` prepares it: turned
/// upright (EXIF), scaled with Pillow's bicubic so its shorter side reaches the target (sides then
/// even), optionally through a smaller size first ("softness"), padded with black to multiples of
/// 16.
public struct SeedVR2Picture {
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int

    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb
        self.width = width
        self.height = height
    }

    /// The image at `path` on an sRGB canvas, transparency over white, upright.
    public init(path: String) throws {
        guard let image = ImagePixels.load(path) else { throw GenerationError.unreadableImage(path) }
        guard image.width >= 16, image.height >= 16 else { throw GenerationError.referenceTooSmall }
        let rgba = ImagePixels.rgbaBytes(image)
        let rgb = ImagePixels.crop(rgba, rgbaWidth: image.width, left: 0, top: 0, width: image.width, height: image.height)
        let upright = Self.upright(rgb, width: image.width, height: image.height, orientation: Self.orientation(path))
        self.init(rgb: upright.rgb, width: upright.width, height: upright.height)
    }

    static func orientation(_ path: String) -> Int {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let value = properties[kCGImagePropertyOrientation] as? NSNumber
        else { return 1 }
        return value.intValue
    }

    /// `ImageOps.exif_transpose`: the pixels as the EXIF orientation says the picture is seen.
    static func upright(_ rgb: [UInt8], width: Int, height: Int, orientation: Int) -> (rgb: [UInt8], width: Int, height: Int) {
        guard (2 ... 8).contains(orientation) else { return (rgb, width, height) }
        let swaps = orientation >= 5
        let (outWidth, outHeight) = swaps ? (height, width) : (width, height)
        var out = [UInt8](repeating: 0, count: rgb.count)
        for y in 0 ..< outHeight {
            for x in 0 ..< outWidth {
                let (sx, sy): (Int, Int) = switch orientation {
                case 2: (width - 1 - x, y)
                case 3: (width - 1 - x, height - 1 - y)
                case 4: (x, height - 1 - y)
                case 5: (y, x)
                case 6: (y, height - 1 - x)
                case 7: (width - 1 - y, height - 1 - x)
                default: (width - 1 - y, x)
                }
                let source = (sy * width + sx) * 3
                let target = (y * outWidth + x) * 3
                out[target] = rgb[source]
                out[target + 1] = rgb[source + 1]
                out[target + 2] = rgb[source + 2]
            }
        }
        return (out, outWidth, outHeight)
    }

    /// The size the upscale comes out at for a scale factor, as mflux's `ScaleFactor` and
    /// `preprocess_image` compute it: the shorter side times the factor, cut down to a multiple of
    /// 16; the other side in proportion; both even.
    public static func outputSize(width: Int, height: Int, factor: Double) -> (width: Int, height: Int) {
        let shorter = Double(min(width, height))
        let product = factor * shorter
        let target = Int(product - product.truncatingRemainder(dividingBy: 16))
        let scale = Double(target) / shorter
        let trueWidth = Int(Double(width) * scale) / 2 * 2
        let trueHeight = Int(Double(height) * scale) / 2 * 2
        return (trueWidth, trueHeight)
    }

    /// The model's input [1, H, W, 3] in [-1, 1] (float32), H and W padded to multiples of 16, and
    /// the size of the picture inside it.
    public func input(width trueWidth: Int, height trueHeight: Int, softness: Double) -> (pixels: MLXArray, rgb: [UInt8], width: Int, height: Int) {
        let factor = 1 + max(0, min(1, softness)) * 7
        var resized: [UInt8]
        if factor > 1 {
            let downWidth = max(2, Int(Double(trueWidth) / factor))
            let downHeight = max(2, Int(Double(trueHeight) / factor))
            let down = PILResample.resize(rgb, width: width, height: height, toWidth: downWidth, toHeight: downHeight, filter: .bicubic)
            resized = PILResample.resize(down, width: downWidth, height: downHeight, toWidth: trueWidth, toHeight: trueHeight, filter: .bicubic)
        } else {
            resized = PILResample.resize(rgb, width: width, height: height, toWidth: trueWidth, toHeight: trueHeight, filter: .bicubic)
        }
        let paddedWidth = trueWidth + (16 - trueWidth % 16) % 16
        let paddedHeight = trueHeight + (16 - trueHeight % 16) % 16
        if paddedWidth != trueWidth || paddedHeight != trueHeight {
            var padded = [UInt8](repeating: 0, count: paddedWidth * paddedHeight * 3)
            for y in 0 ..< trueHeight {
                let row = resized[(y * trueWidth * 3) ..< ((y + 1) * trueWidth * 3)]
                padded.replaceSubrange((y * paddedWidth * 3) ..< (y * paddedWidth * 3 + trueWidth * 3), with: row)
            }
            resized = padded
        }
        let unit = clip(MLXArray(resized, [1, paddedHeight, paddedWidth, 3]).asType(.float32) / Float(255), min: 0, max: 1)
        return (unit * Float(2) - Float(1), resized, paddedWidth, paddedHeight)
    }
}

/// SeedVR2: the VAE, the transformer, the fixed text embedding, one flow step.
public final class SeedVR2Model: FamilyModel {
    public let config: SeedVR2Config
    public let transformer: SeedVR2Transformer
    public let vae: SeedVR2VAE
    /// The positive text embedding SeedVR2 was trained with, [1, 58, 5120] float16.
    public let textEmbedding: MLXArray
    public let bits: Int? = nil
    /// Encoding and decoding always go in 512-pixel tiles, as mflux runs them.
    public var lowRam = false

    public enum SeedVR2Error: LocalizedError {
        case missingFile(String, URL)
        case missingTextEmbedding

        public var errorDescription: String? {
            switch self {
            case .missingFile(let name, let url): "The checkpoint at \(url.path) has no \(name)."
            case .missingTextEmbedding: "The engine is missing SeedVR2's text embedding (seedvr2_pos_emb.safetensors)."
            }
        }
    }

    /// Loads `seedvr2_ema_3b_fp16.safetensors` and `ema_vae_fp16.safetensors` from `modelPath`
    /// (the original checkpoint, as numz/SeedVR2_comfyUI publishes it). The text embedding comes
    /// with the engine unless given.
    public init(modelPath: URL, config: SeedVR2Config, textEmbedding: MLXArray? = nil) throws {
        self.config = config
        transformer = SeedVR2Transformer(config: config)
        vae = SeedVR2VAE(config: config)
        let transformerURL = modelPath.appending(path: config.transformerFile)
        let vaeURL = modelPath.appending(path: config.vaeFile)
        for url in [transformerURL, vaeURL] where !FileManager.default.fileExists(atPath: url.path) {
            throw SeedVR2Error.missingFile(url.lastPathComponent, modelPath)
        }
        try WeightLoading.apply(try loadArrays(url: transformerURL), to: transformer)
        try WeightLoading.apply(SeedVR2VAE.channelsLast(try loadArrays(url: vaeURL)), to: vae)
        if let textEmbedding {
            self.textEmbedding = textEmbedding
        } else {
            guard let url = Self.textEmbeddingURL(), let embedding = try loadArrays(url: url)["embedding"] else {
                throw SeedVR2Error.missingTextEmbedding
            }
            self.textEmbedding = embedding.ndim == 2 ? expandedDimensions(embedding, axis: 0) : embedding
        }
    }

    /// The embedding mflux ships (`seedvr2_text_encoder/embeddings/pos_emb.safetensors`), in the
    /// engine's resource bundle: next to the binary, or in the app's resources.
    static func textEmbeddingURL() -> URL? {
        let bundleName = "TurboEngine_TurboEngineCore.bundle"
        let executable = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        var folders = [executable, executable.deletingLastPathComponent().appending(path: "Resources")]
        if let resources = Bundle.main.resourceURL { folders.append(resources) }
        for folder in folders {
            let url = folder.appending(path: bundleName)
            if let bundle = Bundle(url: url), let file = bundle.url(forResource: "seedvr2_pos_emb", withExtension: "safetensors") {
                return file
            }
            let flat = url.appending(path: "seedvr2_pos_emb.safetensors")
            if FileManager.default.fileExists(atPath: flat.path) { return flat }
        }
        return nil
    }

    // No prompt: the text side is the fixed embedding.
    public func isCached(_ prompt: String) -> Bool { true }
    public func encode(_ prompt: String) throws {}
    public func promptsEncoded() {}

    // MARK: Pipeline

    /// `VAETiler.encode_image_tiled` with mflux's defaults: 512-pixel tiles overlapping by 64,
    /// blended with cosine ramps over the 8 latent pixels of the overlap. `pixels` [1, H, W, 3] →
    /// the latent [1, H/8, W/8, 16].
    public func encodeTiled(_ pixels: MLXArray, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let (height, width) = (pixels.shape[1], pixels.shape[2])
        let tile = 512, overlap = 64, scale = 8
        if height <= tile && width <= tile { return vae.encode(pixels) }
        let latentTile = tile / scale
        let latentOverlap = max(0, min(overlap / scale, latentTile - 1))
        let stride = max(1, latentTile - latentOverlap)
        let latentHeight = (height + scale - 1) / scale
        let latentWidth = (width + scale - 1) / scale
        let ramp = VAETiling.cosRamp(latentOverlap)
        var output = MLXArray.zeros([1, latentHeight, latentWidth, config.latentChannels], dtype: .float32)
        var counts = MLXArray.zeros([1, latentHeight, latentWidth, 1], dtype: .float32)
        for y in Swift.stride(from: 0, to: latentHeight, by: stride) {
            let yEnd = min(y + latentTile, latentHeight)
            for x in Swift.stride(from: 0, to: latentWidth, by: stride) {
                let xEnd = min(x + latentTile, latentWidth)
                if (y > 0 && yEnd - y <= latentOverlap) || (x > 0 && xEnd - x <= latentOverlap) { continue }
                if isCancelled() { throw GenerationError.cancelled }
                let sample = pixels[0..., (y * scale) ..< min(yEnd * scale, height), (x * scale) ..< min(xEnd * scale, width), 0...]
                var encoded = vae.encode(sample).asType(.float32)
                let effectiveHeight = min(yEnd - y, encoded.shape[1], latentHeight - y)
                let effectiveWidth = min(xEnd - x, encoded.shape[2], latentWidth - x)
                encoded = encoded[0..., 0 ..< effectiveHeight, 0 ..< effectiveWidth, 0...]
                let ovH = max(0, min(latentOverlap, effectiveHeight - 1))
                let ovW = max(0, min(latentOverlap, effectiveWidth - 1))
                var wh = [Float](repeating: 1, count: effectiveHeight)
                var ww = [Float](repeating: 1, count: effectiveWidth)
                if ovH > 0 {
                    if y > 0 { for i in 0 ..< ovH { wh[i] = ramp[i] } }
                    if yEnd < latentHeight { for i in 0 ..< ovH { wh[effectiveHeight - ovH + i] = 1 - ramp[i] } }
                }
                if ovW > 0 {
                    if x > 0 { for i in 0 ..< ovW { ww[i] = ramp[i] } }
                    if xEnd < latentWidth { for i in 0 ..< ovW { ww[effectiveWidth - ovW + i] = 1 - ramp[i] } }
                }
                let weights = MLXArray(wh, [1, effectiveHeight, 1, 1]) * MLXArray(ww, [1, 1, effectiveWidth, 1])
                let rows = y ..< (y + effectiveHeight)
                let columns = x ..< (x + effectiveWidth)
                output[0..., rows, columns, 0...] = output[0..., rows, columns, 0...] + encoded * weights
                counts[0..., rows, columns, 0...] = counts[0..., rows, columns, 0...] + weights
                eval(output, counts)
            }
        }
        return output / maximum(counts, MLXArray(Float(1e-6)))
    }

    /// The noise mflux draws, channels first as it draws it: [1, 16, 1, h, w].
    public static func noise(seed: Int, latentHeight: Int, latentWidth: Int, channels: Int = 16) -> MLXArray {
        MLXRandom.normal([1, channels, 1, latentHeight, latentWidth], key: MLXRandom.key(UInt64(seed)))
    }

    /// The transformer's input: the noise, the picture's latent (channels last, [1, h, w, 16]) and
    /// a mask of ones, channels first [1, 33, 1, h, w].
    public static func modelInput(noise: MLXArray, latent: MLXArray) -> MLXArray {
        let condition = expandedDimensions(latent.transposed(0, 3, 1, 2), axis: 2)
        let mask = MLXArray.ones([1, 1, 1, latent.shape[1], latent.shape[2]])
        return concatenated([noise, concatenated([condition, mask], axis: 1)], axis: 1)
    }

    /// `SeedVR2EulerScheduler.step` over `steps` equal steps of [1000, 0].
    public static func step(latents: MLXArray, flow: MLXArray, index: Int, steps: Int) -> MLXArray {
        let total: Float = 1000
        let stepSize = total / Float(steps)
        let t = max(total - Float(index) * stepSize, 0)
        let s = max(total - Float(index + 1) * stepSize, 0)
        let tNorm = MLXArray(t) / total
        let sNorm = MLXArray(s) / total
        let predictedClean = latents - tNorm * flow
        let predictedNoise = latents + (1 - tNorm) * flow
        return s > 0 ? (1 - sNorm) * predictedClean + sNorm * predictedNoise : predictedClean
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        guard let path = request.imagePath else { throw GenerationError.referenceRequired }
        let picture = try SeedVR2Picture(path: path)
        let size = SeedVR2Picture.outputSize(width: picture.width, height: picture.height, factor: request.upscale ?? 2)
        guard size.width >= 16, size.height >= 16 else { throw GenerationError.sizeTooSmall }
        return try upscale(picture: picture, width: size.width, height: size.height, softness: request.softness ?? 0,
                           seed: request.seed, phase: phase, progress: progress, isCancelled: isCancelled)
    }

    /// The whole upscale of `picture` to `width` × `height` (even sides).
    public func upscale(
        picture: SeedVR2Picture, width: Int, height: Int, softness: Double, seed: Int,
        phase: (GenerationPhase) -> Void, progress: (Int, Int) -> Void, isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        phase(.encoding)
        let input = picture.input(width: width, height: height, softness: softness)
        let latent = try encodeTiled(input.pixels, isCancelled: isCancelled)
        eval(latent)
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let steps = 1
        var latents = Self.noise(seed: seed, latentHeight: latent.shape[1], latentWidth: latent.shape[2], channels: config.latentChannels)
        let condition = Self.modelInput(noise: latents, latent: latent)[0..., config.latentChannels...]
        for index in 0 ..< steps {
            let flow = try transformer.forward(vid: concatenated([latents, condition], axis: 1), txt: textEmbedding,
                                               timestep: 1000 - Float(index) * 1000 / Float(steps), isCancelled: isCancelled)
            latents = Self.step(latents: latents, flow: flow, index: index, steps: steps)
            eval(latents)
            progress(index + 1, steps)
            if isCancelled() { throw GenerationError.cancelled }
        }

        phase(.decoding)
        let grid = latents[0..., 0..., 0].transposed(0, 2, 3, 1)
        let decoded = try VAETiling.decode(grid, decode: { vae.decode($0) }, isCancelled: isCancelled)
        let content = decoded[0..., 0 ..< height, 0 ..< width, 0...].asType(.float32)
        let style = input.pixels[0..., 0 ..< height, 0 ..< width, 0...]
        eval(content, style)
        let corrected = SeedVR2ColorCorrection.apply(content: content.asArray(Float.self), style: style.asArray(Float.self),
                                                     width: width, height: height)
        let pixels = Pixels.toPixels(MLXArray(corrected, [1, height, width, 3]))
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }
}
