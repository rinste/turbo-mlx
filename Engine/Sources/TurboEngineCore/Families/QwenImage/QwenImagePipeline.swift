import Foundation
import MLX
import MLXNN

/// Qwen-Image 2512: the Qwen2.5-VL encoder, the transformer and the 3D decoder, with the prompt
/// cache and the loop of mflux's `QwenImage.generate_image` (true classifier-free guidance, the
/// linear schedule). The 7B encoder is kept out of memory while the 20B transformer works in
/// low-RAM mode, and vice versa: each side reloads lazily when it is next needed.
public final class QwenImageModel: PreviewingFamilyModel {
    public let config: QwenImageConfig
    public let modelPath: URL
    public private(set) var bits: Int?
    public var lowRam = false

    private var checkpoint: Checkpoint
    private var prompter: TemplatePrompter?
    private var textEncoder: Qwen25TextEncoder?
    private var imageSide: (transformer: QwenImageTransformer, vae: QwenImageVAE)?
    /// The transformer has generated: its weights are resident rather than lazy.
    private var imageSideUsed = false
    /// Prompt embeddings per prompt, [1, T, joint dim] bf16.
    private var promptCache: [String: MLXArray] = [:]
    private var negativeEmbeds: MLXArray?

    /// Loads the checkpoint at `modelPath` (mflux format). Weights stay lazy until first use.
    public init(modelPath: URL, config: QwenImageConfig, loadTokenizer: Bool = true) throws {
        self.config = config
        self.modelPath = modelPath
        checkpoint = Checkpoint(root: modelPath)
        try loadImageSide()
        try loadTextSide()
        if loadTokenizer {
            prompter = try TemplatePrompter(
                folder: modelPath.appending(path: "tokenizer", directoryHint: .isDirectory),
                template: QwenImageConfig.template, maxLength: config.maxSequenceLength, addSpecialTokens: true
            )
        }
    }

    private func loadTextSide() throws {
        let encoder = Qwen25TextEncoder(config: config.textEncoder)
        try WeightLoading.apply(try checkpoint.loadComponent("text_encoder"), to: encoder, ignoring: Qwen25TextEncoder.ignoresKey)
        textEncoder = encoder
    }

    private func loadImageSide() throws {
        let transformer = QwenImageTransformer(config: config.transformer)
        let vae = QwenImageVAE(outChannels: 3, baseDim: config.vaeBaseDim, normalization: .meanStd)
        try WeightLoading.apply(try checkpoint.loadComponent("transformer"), to: transformer)
        try WeightLoading.apply(QwenImageVAE.weights(try checkpoint.loadComponent("vae")), to: vae, ignoring: QwenImageVAE.ignoresKey)
        bits = checkpoint.bits
        imageSide = (transformer, vae)
        imageSideUsed = false
    }

    /// The text encoder, reloaded if it had been released. In low-RAM mode a resident transformer
    /// is released first, so the two are never in memory together (mflux reloads the whole model
    /// for the same reason).
    public func loadedTextEncoder() throws -> Qwen25TextEncoder {
        if let textEncoder { return textEncoder }
        if lowRam, imageSideUsed {
            imageSide = nil
            Memory.clearCache()
        }
        try loadTextSide()
        return textEncoder!
    }

    /// The transformer and the decoder, reloaded (lazily) if they had been released.
    public func loadedImageSide() throws -> (transformer: QwenImageTransformer, vae: QwenImageVAE) {
        if let imageSide { return imageSide }
        try loadImageSide()
        return imageSide!
    }

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil && negativeEmbeds != nil }

    public func encode(_ prompt: String) throws {
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        if promptCache[prompt] == nil {
            promptCache[prompt] = try promptEmbeds(ids: prompter.tokenIds(prompt))
        }
        if negativeEmbeds == nil {
            negativeEmbeds = try promptEmbeds(ids: prompter.tokenIds(QwenImageConfig.negativePrompt))
        }
    }

    public func promptsEncoded() {
        if lowRam, textEncoder != nil {
            textEncoder = nil
            Memory.clearCache()
        }
    }

    /// The transformer's text input for tokenized text: [1, T − dropIndex, joint dim] bf16.
    public func promptEmbeds(ids: [Int]) throws -> MLXArray {
        let inputIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let embeds = try loadedTextEncoder().promptEmbeds(inputIds: inputIds)
        eval(embeds)
        return embeds
    }

    // MARK: Sampling

    /// The initial noise: [1, (H/16)·(W/16), 64] in float32, as mflux's FLUX-style creator makes it.
    public static func initialLatents(width: Int, height: Int, seed: Int) -> MLXArray {
        MLXRandom.normal([1, (height / 16) * (width / 16), 64], key: MLXRandom.key(UInt64(seed)))
    }

    /// `compute_guided_noise`: the CFG combination rescaled to the conditional prediction's norm.
    /// In float32 whatever the stream's precision (the norms' ratio is a fine quantity), returned
    /// in the prediction's.
    public static func guidedNoise(_ noise: MLXArray, negative: MLXArray, guidance: Float) -> MLXArray {
        let positive32 = noise.asType(.float32)
        let negative32 = negative.asType(.float32)
        let combined = negative32 + guidance * (positive32 - negative32)
        let condNorm = sqrt((positive32 * positive32).sum(axis: -1, keepDims: true) + 1e-12)
        let noiseNorm = sqrt((combined * combined).sum(axis: -1, keepDims: true) + 1e-12)
        return (combined * (condNorm / noiseNorm)).asType(noise.dtype)
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        preview: (Preview) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        let width = 16 * (request.width / 16)
        let height = 16 * (request.height / 16)
        guard width >= 16, height >= 16 else { throw GenerationError.sizeTooSmall }

        phase(.encoding)
        try encode(request.prompt)
        guard let prompt = promptCache[request.prompt], let negative = negativeEmbeds else { throw GenerationError.cancelled }
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let (transformer, vae) = try loadedImageSide()
        let (latentHeight, latentWidth) = (height / 16, width / 16)
        let schedule = LinearSchedule(steps: request.steps, width: width, height: height, shift: config.shift)
        let guidance = Float(request.guidance)
        var latents = Self.initialLatents(width: width, height: height, seed: request.seed)
        // The reference keeps the latents, and with them the stream, in float32; the 16-bit option
        // starts them in bf16, which the transformer follows (the decode stays in float32).
        if request.halfPrecision { latents = latents.asType(.bfloat16) }
        for t in 0 ..< request.steps {
            let sigma = schedule.sigmas[t]
            var noise = transformer(latents: latents, prompt: prompt, timestep: sigma, latentHeight: latentHeight, latentWidth: latentWidth)
            // At guidance 1 the unconditional pass would cancel out exactly: skip it.
            if guidance > 1 {
                let negativeNoise = transformer(latents: latents, prompt: negative, timestep: sigma, latentHeight: latentHeight, latentWidth: latentWidth)
                noise = Self.guidedNoise(noise, negative: negativeNoise, guidance: guidance)
            }
            let glimpse = LatentPreview.shows(step: t + 1, of: request.steps)
                ? LatentPreview.predicted(latents: latents, noise: noise, sigma: sigma) : nil
            latents = schedule.step(latents: latents, noise: noise, index: t)
            eval(latents)
            progress(t + 1, request.steps)
            if let glimpse {
                preview(Preview(pixels: Self.previewPixels(latents: glimpse, latentHeight: latentHeight, latentWidth: latentWidth, vae: vae), step: t + 1))
            }
            if isCancelled() { throw GenerationError.cancelled }
        }
        imageSideUsed = true

        phase(.decoding)
        let pixels = try decode(latents: latents.asType(.float32), latentHeight: latentHeight, latentWidth: latentWidth, vae: vae, isCancelled: isCancelled)
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }

    /// A small image of what packed latents [1, h·w, 64] (a step's prediction) hold: the grid
    /// pooled to about 384 pixels on the longer side, decoded in one piece, in float32 as the image.
    public static func previewPixels(latents: MLXArray, latentHeight: Int, latentWidth: Int, vae: QwenImageVAE) -> MLXArray {
        let grid = unpack(latents, latentHeight: latentHeight, latentWidth: latentWidth).asType(.float32)
        let factor = LatentPreview.factor(height: grid.shape[1], width: grid.shape[2], scale: QwenImageVAE.spatialScale)
        return Pixels.toPixels(vae.decode(LatentPreview.pooled(grid, factor: factor)))
    }

    /// Packed latents [1, h·w, 64] → [H, W, 3] uint8 pixels; in tiles with Save memory on.
    public func decode(latents: MLXArray, latentHeight: Int, latentWidth: Int, vae: QwenImageVAE, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        try Self.decode(latents: latents, latentHeight: latentHeight, latentWidth: latentWidth, vae: vae, tiled: lowRam, isCancelled: isCancelled)
    }

    /// The same for any Qwen-Image checkpoint (the edit model decodes the same way).
    public static func decode(latents: MLXArray, latentHeight: Int, latentWidth: Int, vae: QwenImageVAE, tiled: Bool, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let grid = unpack(latents, latentHeight: latentHeight, latentWidth: latentWidth)
        let decoded = tiled
            ? try VAETiling.decode(grid, spatialScale: QwenImageVAE.spatialScale, decode: { vae.decode($0) }, isCancelled: isCancelled)
            : vae.decode(grid)
        return Pixels.toPixels(decoded)
    }

    /// `unpack_latents`, then channels last: [1, h·w, 64] → [1, 2h, 2w, 16].
    public static func unpack(_ latents: MLXArray, latentHeight: Int, latentWidth: Int) -> MLXArray {
        let unpacked = latents.reshaped([1, latentHeight, latentWidth, 16, 2, 2])
            .transposed(0, 3, 1, 4, 2, 5)
            .reshaped([1, 16, latentHeight * 2, latentWidth * 2])
        return unpacked.transposed(0, 2, 3, 1)
    }
}
