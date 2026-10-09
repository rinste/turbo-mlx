import Foundation
import MLX
import MLXNN

/// The picture an edit starts from, as mflux reads it (`ImageUtil.load_image(...).convert("RGB")`):
/// its bytes, then resized for each of its two uses the way Pillow resizes.
public struct QwenEditPicture {
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int

    /// CLIP's normalization, which Qwen2.5-VL's image processor applies.
    static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    static let std: [Float] = [0.26862954, 0.26130258, 0.27577711]

    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb
        self.width = width
        self.height = height
    }

    /// The image at `path` on an sRGB canvas, transparency over white.
    public init(path: String) throws {
        guard let image = ImagePixels.load(path) else { throw GenerationError.unreadableImage(path) }
        guard image.width >= 16, image.height >= 16 else { throw GenerationError.referenceTooSmall }
        let rgba = ImagePixels.rgbaBytes(image)
        self.init(rgb: ImagePixels.crop(rgba, rgbaWidth: image.width, left: 0, top: 0, width: image.width, height: image.height),
                  width: image.width, height: image.height)
    }

    /// What the vision tower reads, as `QwenVisionLanguageTokenizer.tokenize_with_image` and
    /// `QwenImageProcessor` prepare it: about 384 × 384 pixels in the picture's proportions (sides
    /// multiples of 32), then sides multiples of 28 (`smart_resize`), both bicubic; normalized; in
    /// 14-pixel patches of two identical frames, each 2 × 2 patches in a row. Returns the patches
    /// [N, 3·2·14·14] in float32 and their grid.
    public func visionInput(_ vision: QwenImageConfig.Vision) -> (pixelValues: MLXArray, grid: QwenVisionGrid, resized: (rgb: [UInt8], width: Int, height: Int)) {
        let condition = Self.conditionSize(width: width, height: height)
        let first = PILResample.resize(rgb, width: width, height: height, toWidth: condition.width, toHeight: condition.height, filter: .bicubic)
        let patch = vision.patchSize
        let merge = vision.spatialMergeSize
        let target = Self.smartResize(height: condition.height, width: condition.width, factor: patch * merge,
                                      minPixels: 56 * 56, maxPixels: 28 * 28 * 1280)
        let bytes = PILResample.resize(first, width: condition.width, height: condition.height,
                                       toWidth: target.width, toHeight: target.height, filter: .bicubic)
        // numpy's float32 arithmetic: divided by 255, minus the mean, over the deviation.
        var values = [Float](repeating: 0, count: bytes.count)
        for index in 0 ..< bytes.count {
            let channel = index % 3
            values[index] = (Float(bytes[index]) / 255 - Self.mean[channel]) / Self.std[channel]
        }
        let image = MLXArray(values, [target.height, target.width, 3]).transposed(2, 0, 1)
        let frames = vision.temporalPatchSize
        let (rows, columns) = (target.height / patch, target.width / patch)
        let patches = stacked(Array(repeating: image, count: frames), axis: 0)
            .reshaped([1, frames, 3, rows / merge, merge, patch, columns / merge, merge, patch])
            .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshaped([rows * columns, 3 * frames * patch * patch])
        return (patches, QwenVisionGrid(t: 1, h: rows, w: columns), (bytes, target.width, target.height))
    }

    /// What the VAE encodes (`ImageUtil.scale_to_dimensions` then `to_array`): resized with Lanczos
    /// to `width` × `height` unless it is that size already, [1, H, W, 3] float32 in [-1, 1].
    public func vaeInput(width: Int, height: Int) -> MLXArray {
        let bytes = width == self.width && height == self.height
            ? rgb
            : PILResample.resize(rgb, width: self.width, height: self.height, toWidth: width, toHeight: height, filter: .lanczos)
        let unit = MLXArray(bytes.map { Float($0) / 255 }, [1, height, width, 3])
        return Float(2) * unit - Float(1)
    }

    /// About 384 × 384 pixels in the picture's proportions, sides rounded (half to even, as Python
    /// rounds) to multiples of 32.
    public static func conditionSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let ratio = Double(width) / Double(height)
        let conditionWidth = (Double(384 * 384) * ratio).squareRoot()
        let conditionHeight = conditionWidth / ratio
        return (Int((conditionWidth / 32).rounded(.toNearestOrEven)) * 32, Int((conditionHeight / 32).rounded(.toNearestOrEven)) * 32)
    }

    /// Qwen2.5-VL's `smart_resize`: sides multiples of `factor`, the area kept within bounds.
    public static func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int) -> (height: Int, width: Int) {
        var h = Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor
        var w = Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor
        if h * w > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (h, w)
    }
}

/// Qwen-Image-Edit 2511: an image made from a picture as the prompt says, after mflux's
/// `QwenImageEdit.generate_image`. The picture goes in twice: through Qwen2.5-VL's vision tower
/// into the prompt (about 384 × 384 pixels), and through the VAE's encoder into latents that
/// follow the image's in every pass. The transformer, the schedule, the guidance and the decoder
/// are Qwen-Image's. As there, the 7B encoder and the 20B transformer are never in memory together
/// in low-RAM mode.
public final class QwenImageEditModel: PreviewingFamilyModel, LoRAAdaptable {
    public let config: QwenImageConfig
    public let modelPath: URL
    public private(set) var bits: Int?
    public var lowRam = false

    private var checkpoint: Checkpoint
    private var prompter: TemplatePrompter?
    private var textEncoder: Qwen25TextEncoder?
    private var imageSide: (transformer: QwenImageTransformer, vae: QwenImageVAE)?
    /// On the transformer, and put back on it whenever it is loaded again (Save memory releases it).
    public let loras = LoRAAdapters(table: .qwen)
    public var adaptedModule: Module? { imageSide?.transformer }
    /// The transformer has generated: its weights are resident rather than lazy.
    private var imageSideUsed = false
    /// The prompt's and the negative prompt's embeddings [1, T, hidden] in float16, per picture
    /// and prompt, the most recent last.
    private var embedsCache: [(key: String, prompt: MLXArray, negative: MLXArray)] = []
    private static let cachedEdits = 8

    /// Loads the checkpoint at `modelPath` (mflux format). Weights stay lazy until first use.
    public init(modelPath: URL, config: QwenImageConfig, loadTokenizer: Bool = true) throws {
        precondition(config.vision != nil, "an edit model needs the vision tower's geometry")
        self.config = config
        self.modelPath = modelPath
        checkpoint = Checkpoint(root: modelPath)
        try loadImageSide()
        try loadTextSide()
        if loadTokenizer {
            // The whole text is built here (the picture's placeholders depend on its size).
            prompter = try TemplatePrompter(
                folder: modelPath.appending(path: "tokenizer", directoryHint: .isDirectory),
                template: "{}", maxLength: nil, addSpecialTokens: true
            )
        }
    }

    private func loadTextSide() throws {
        let encoder = Qwen25TextEncoder(config: config.textEncoder, vision: config.vision)
        try WeightLoading.apply(try checkpoint.loadComponent("text_encoder"), to: encoder, ignoring: Qwen25TextEncoder.ignoresKeyWithVision)
        textEncoder = encoder
    }

    private func loadImageSide() throws {
        let transformer = QwenImageTransformer(config: config.transformer)
        let vae = QwenImageVAE(outChannels: 3, baseDim: config.vaeBaseDim, normalization: .meanStd, withEncoder: true)
        try WeightLoading.apply(try checkpoint.loadComponent("transformer"), to: transformer)
        try WeightLoading.apply(QwenImageVAE.weights(try checkpoint.loadComponent("vae")), to: vae, ignoring: QwenImageVAE.ignoresKeyWithEncoder)
        try loras.reapply(on: transformer)
        bits = checkpoint.bits
        imageSide = (transformer, vae)
        imageSideUsed = false
    }

    /// The text encoder, reloaded if it had been released (a resident transformer goes first in
    /// low-RAM mode).
    public func loadedTextEncoder() throws -> Qwen25TextEncoder {
        if let textEncoder { return textEncoder }
        if lowRam, imageSideUsed {
            imageSide = nil
            Memory.clearCache()
        }
        try loadTextSide()
        return textEncoder!
    }

    /// The transformer and the autoencoder, reloaded (lazily) if they had been released.
    public func loadedImageSide() throws -> (transformer: QwenImageTransformer, vae: QwenImageVAE) {
        if let imageSide { return imageSide }
        try loadImageSide()
        return imageSide!
    }

    // An edit's prompt is encoded with its picture, in `generate`: nothing to do beforehand.
    public func isCached(_ prompt: String) -> Bool { true }
    public func encode(_ prompt: String) throws {}
    public func promptsEncoded() {}

    /// The template's token ids for `prompt` with `imageTokens` placeholders for the picture.
    public func tokenIds(prompt: String, imageTokens: Int) throws -> [Int] {
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        return prompter.tokenIds(QwenImageConfig.editText(prompt: prompt, imageTokens: imageTokens))
    }

    /// The prompt's and the negative prompt's embeddings with `picture` (identified by `key`), the
    /// negative mflux's empty one unless `negativePrompt` is given. The reference reads the picture
    /// once for each; its tokens are the same both times, read once here.
    public func embeds(prompt: String, negativePrompt: String? = nil, picture: QwenEditPicture, key: String) throws
        -> (prompt: MLXArray, negative: MLXArray) {
        let negativeText = negativePrompt ?? QwenImageConfig.editNegativePrompt
        let cacheKey = [key, prompt, negativeText].joined(separator: "\u{1F}")
        if let cached = embedsCache.first(where: { $0.key == cacheKey }) { return (cached.prompt, cached.negative) }
        let encoder = try loadedTextEncoder()
        guard let vision = config.vision else { preconditionFailure("an edit model needs its vision tower") }
        let input = picture.visionInput(vision)
        let image = encoder.imageEmbeds(pixelValues: input.pixelValues, grids: [input.grid])
        eval(image)
        let tokens = input.grid.patches / (vision.spatialMergeSize * vision.spatialMergeSize)
        func run(_ text: String) throws -> MLXArray {
            let ids = try tokenIds(prompt: text, imageTokens: tokens)
            let embeds = encoder.editEmbeds(inputIds: MLXArray(ids.map { Int32($0) }, [1, ids.count]), imageEmbeds: image)
            eval(embeds)
            return embeds
        }
        let result = (prompt: try run(prompt), negative: try run(negativeText))
        embedsCache.append((cacheKey, result.prompt, result.negative))
        if embedsCache.count > Self.cachedEdits { embedsCache.removeFirst() }
        return result
    }

    /// The size the picture is encoded at: the image's own when the proportions are the same (mflux
    /// resizes the picture to the image's size), otherwise about as many pixels as the image in the
    /// picture's proportions, sides multiples of 16, rather than stretching it (the official
    /// pipeline keeps them too).
    public static func referenceSize(pictureWidth: Int, pictureHeight: Int, width: Int, height: Int) -> (width: Int, height: Int) {
        if pictureWidth * height == pictureHeight * width { return (width, height) }
        let area = Double(width * height)
        let ratio = Double(pictureWidth) / Double(pictureHeight)
        return (max(16, Int(((area * ratio).squareRoot() / 16).rounded()) * 16),
                max(16, Int(((area / ratio).squareRoot() / 16).rounded()) * 16))
    }

    /// A picture file's identity for the cache: its path, size and modification date.
    static func pictureKey(_ path: String) -> String {
        let attributes = try? FileManager.default.attributesOfItem(atPath: path)
        let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
        let modified = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return "\(path)|\(size)|\(modified)"
    }

    /// The picture's latents as they follow the image's tokens: [1, h·w, 64].
    public func referenceLatents(picture: QwenEditPicture, width: Int, height: Int, vae: QwenImageVAE) -> MLXArray {
        let latents = QwenImageVAE.pack(vae.encode(picture.vaeInput(width: width, height: height)))
        eval(latents)
        return latents
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        preview: (Preview) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        guard let path = request.imagePath else { throw GenerationError.referenceRequired }
        let width = 16 * (request.width / 16)
        let height = 16 * (request.height / 16)
        guard width >= 16, height >= 16 else { throw GenerationError.sizeTooSmall }

        phase(.encoding)
        let picture = try QwenEditPicture(path: path)
        let (prompt, negative) = try embeds(prompt: request.prompt, negativePrompt: request.negativePrompt, picture: picture,
                                            key: Self.pictureKey(path))
        if lowRam, textEncoder != nil {
            textEncoder = nil
            Memory.clearCache()
        }
        if isCancelled() { throw GenerationError.cancelled }

        let (transformer, vae) = try loadedImageSide()
        let reference = Self.referenceSize(pictureWidth: picture.width, pictureHeight: picture.height, width: width, height: height)
        let referenceTokens = referenceLatents(picture: picture, width: reference.width, height: reference.height, vae: vae)
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let grids = [(height: height / 16, width: width / 16), (height: reference.height / 16, width: reference.width / 16)]
        let schedule = LinearSchedule(steps: request.steps, width: width, height: height, shift: config.shift)
        let guidance = Float(request.guidance)
        var latents = QwenImageModel.initialLatents(width: width, height: height, seed: request.seed)
        // As in Qwen-Image: the 16-bit option starts the latents, and the picture's, in bf16.
        var pictureTokens = referenceTokens
        if request.halfPrecision {
            latents = latents.asType(.bfloat16)
            pictureTokens = pictureTokens.asType(.bfloat16)
        }
        let imageTokens = latents.shape[1]
        for t in 0 ..< request.steps {
            let sigma = schedule.sigmas[t]
            let input = concatenated([latents, pictureTokens], axis: 1)
            var noise = transformer(latents: input, prompt: prompt, timestep: sigma, grids: grids)[0..., 0 ..< imageTokens]
            // At guidance 1 the unconditional pass would cancel out exactly: skip it.
            if guidance > 1 {
                let negativeNoise = transformer(latents: input, prompt: negative, timestep: sigma, grids: grids)[0..., 0 ..< imageTokens]
                noise = QwenImageModel.guidedNoise(noise, negative: negativeNoise, guidance: guidance)
            }
            let glimpse = LatentPreview.shows(step: t + 1, of: request.steps)
                ? LatentPreview.predicted(latents: latents, noise: noise, sigma: sigma) : nil
            latents = schedule.step(latents: latents, noise: noise, index: t)
            eval(latents)
            progress(t + 1, request.steps)
            if let glimpse {
                preview(Preview(pixels: QwenImageModel.previewPixels(latents: glimpse, latentHeight: height / 16, latentWidth: width / 16, vae: vae), step: t + 1))
            }
            if isCancelled() { throw GenerationError.cancelled }
        }
        imageSideUsed = true

        phase(.decoding)
        let pixels = try QwenImageModel.decode(latents: latents.asType(.float32), latentHeight: height / 16, latentWidth: width / 16,
                                               vae: vae, tiled: lowRam, isCancelled: isCancelled)
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }
}
