import Foundation
import MLX
import MLXNN

/// Z-Image Turbo: the Qwen3 encoder, the S3-DiT and the decoder, with the prompt cache and the
/// sampling loop of mflux's `ZImage.generate_image` (guidance off, the linear schedule).
public final class ZImageModel: PreviewingFamilyModel, LoRAAdaptable {
    public let config: ZImageConfig
    public let modelPath: URL
    public let textEncoder: Qwen3TextEncoder
    public let transformer: S3DiTTransformer
    public let vae: ZImageVAE
    public private(set) var bits: Int?
    /// Save memory: decode in tiles. The Qwen3 4B encoder is small enough to stay resident.
    public var lowRam = false
    private var prompter: Qwen3Prompter?
    /// Caption features per prompt: [tokens, capFeatDim] bf16.
    private var promptCache: [String: MLXArray] = [:]
    public let loras = LoRAAdapters(table: .zImage)
    public var adaptedModule: Module? { transformer }

    /// Loads the checkpoint at `modelPath` (mflux format). Weights stay lazy until first use.
    public init(modelPath: URL, config: ZImageConfig, loadTokenizer: Bool = true) throws {
        self.config = config
        self.modelPath = modelPath
        textEncoder = Qwen3TextEncoder(config: config.textEncoder)
        transformer = S3DiTTransformer(config: config.transformer)
        vae = ZImageVAE(blockOutChannels: config.vaeBlockOutChannels)

        var checkpoint = Checkpoint(root: modelPath)
        try WeightLoading.apply(try checkpoint.loadComponent("text_encoder"), to: textEncoder, ignoring: Qwen3TextEncoder.ignoresKey)
        try WeightLoading.apply(try checkpoint.loadComponent("transformer"), to: transformer)
        try WeightLoading.apply(try checkpoint.loadComponent("vae"), to: vae, ignoring: ZImageVAE.ignoresKey)
        bits = checkpoint.bits
        if loadTokenizer {
            prompter = try Qwen3Prompter(modelPath: modelPath, maxLength: config.maxSequenceLength, enableThinking: true)
        }
    }

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil }

    public func encode(_ prompt: String) throws {
        _ = try encodePrompt(prompt)
    }

    public func promptsEncoded() {}

    /// The caption features of a prompt (once; later calls read the cache).
    @discardableResult
    public func encodePrompt(_ prompt: String) throws -> MLXArray {
        if let cached = promptCache[prompt] { return cached }
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        // The encoder is causal and the padding is masked, so the real tokens' states are the
        // same without the padding mflux adds to 512; only they are read (`cap_feats[0, :num_valid]`).
        let ids = try prompter.tokenIds(prompt)
        let inputIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let encoded = encode(inputIds: inputIds)
        promptCache[prompt] = encoded
        return encoded
    }

    /// The second-to-last hidden state of the Qwen3 encoder run in float32, as bf16: [tokens, capFeatDim].
    public func encode(inputIds: MLXArray) -> MLXArray {
        let mask = MLXArray.ones(inputIds.shape, dtype: .int32)
        let layer = config.textEncoder.numHiddenLayers - 1
        let states = textEncoder.hiddenStates(inputIds: inputIds, attentionMask: mask, computeType: .float32, through: layer)
        let features = states[layer].asType(modelPrecision)[0]
        eval(features)
        return features
    }

    // MARK: Sampling

    /// The initial noise for a size and seed: [16, 1, H/8, W/8] in the model's precision.
    public static func initialLatents(width: Int, height: Int, seed: Int) -> MLXArray {
        MLXRandom.normal([ZImageVAE.latentChannels, 1, height / 8, width / 8], key: MLXRandom.key(UInt64(seed)))
            .asType(modelPrecision)
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
        let capFeats = try encodePrompt(request.prompt)
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        // The reference lets its float32 timestep embedding promote the stream to float32; the
        // 16-bit option keeps it in the weights' precision, as Ming-Image's S3-DiT always does.
        transformer.setKeepsWeightsPrecision(request.halfPrecision)
        let schedule = LinearSchedule(steps: request.steps, width: width, height: height, shift: config.shift)
        var latents = Self.initialLatents(width: width, height: height, seed: request.seed)
        for t in 0 ..< request.steps {
            let sigma = schedule.sigmas[t]
            let noise = transformer(latents: latents, timestep: MLXArray([1 - sigma]), capFeats: capFeats)
            let glimpse = LatentPreview.shows(step: t + 1, of: request.steps)
                ? LatentPreview.predicted(latents: latents, noise: noise, sigma: sigma) : nil
            latents = schedule.step(latents: latents, noise: noise, index: t)
            eval(latents)
            progress(t + 1, request.steps)
            if let glimpse { preview(Preview(pixels: previewPixels(latents: glimpse), step: t + 1)) }
            if isCancelled() { throw GenerationError.cancelled }
        }

        phase(.decoding)
        let pixels = try decode(latents: latents, isCancelled: isCancelled)
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }

    /// A small image of what `latents` ([16, 1, h, w], a step's prediction) hold: the grid pooled
    /// to about 384 pixels on the longer side, decoded in one piece.
    func previewPixels(latents: MLXArray) -> MLXArray {
        let grid = latents.reshaped([1, latents.shape[0], latents.shape[2], latents.shape[3]]).transposed(0, 2, 3, 1)
        let factor = LatentPreview.factor(height: grid.shape[1], width: grid.shape[2], scale: ZImageVAE.spatialScale)
        return Pixels.toPixels(vae.decode(LatentPreview.pooled(grid, factor: factor)))
    }

    /// [16, 1, h, w] latents → [H, W, 3] uint8 pixels; in tiles with Save memory on.
    public func decode(latents: MLXArray, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let grid = latents.reshaped([1, latents.shape[0], latents.shape[2], latents.shape[3]]).transposed(0, 2, 3, 1)
        let decoded = lowRam
            ? try VAETiling.decode(grid, spatialScale: ZImageVAE.spatialScale, decode: { vae.decode($0) }, isCancelled: isCancelled)
            : vae.decode(grid)
        return Pixels.toPixels(decoded)
    }
}
