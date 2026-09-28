import Foundation
import MLX
import MLXNN

/// Ming-Image 0.1 Design: the Ling MoE encoder with its connector and heads, the S3-DiT and the
/// RGBA decoder, with the prompt cache and the loop of mflux's `MingImage.generate_image` (the
/// static-shift schedule, guidance against zeroed conditions). The ~16B-parameter text side is
/// kept out of memory while the DiT works in low-RAM mode, and vice versa.
public final class MingModel: FamilyModel {
    /// mflux keeps the last 16 prompts.
    static let promptCacheSize = 16

    public let config: MingConfig
    public let modelPath: URL
    public private(set) var bits: Int?
    public var lowRam = false

    private var checkpoint: Checkpoint
    private var prompter: TemplatePrompter?
    private var textSide: MingConditionEncoder?
    private var imageSide: (transformer: S3DiTTransformer, vae: QwenImageVAE)?
    private var imageSideUsed = false
    /// (capFeats, capFeats2) per prompt, in the order they were encoded.
    private var promptCache: [(prompt: String, features: (MLXArray, MLXArray))] = []

    /// Loads the checkpoint at `modelPath` (mflux format). Weights stay lazy until first use.
    public init(modelPath: URL, config: MingConfig, loadTokenizer: Bool = true) throws {
        self.config = config
        self.modelPath = modelPath
        checkpoint = Checkpoint(root: modelPath)
        try loadImageSide()
        try loadTextSide()
        if loadTokenizer {
            prompter = try TemplatePrompter(
                folder: modelPath.appending(path: "mllm", directoryHint: .isDirectory),
                template: MingConfig.promptTemplate, maxLength: nil, addSpecialTokens: false
            )
        }
    }

    private func loadTextSide() throws {
        let encoder = LingMoeEncoder(config: config.encoder)
        let connector = MingConnector(config: config.connector)
        let heads = MingHeads(config: config)
        try WeightLoading.apply(try checkpoint.loadComponent("mllm"), to: encoder)
        try WeightLoading.apply(try checkpoint.loadComponent("connector"), to: connector)
        try WeightLoading.apply(try checkpoint.loadComponent("mlp"), to: heads)
        textSide = MingConditionEncoder(config: config, encoder: encoder, connector: connector, heads: heads)
    }

    private func loadImageSide() throws {
        let transformer = S3DiTTransformer(config: config.transformer)
        let vae = QwenImageVAE(outChannels: 4, baseDim: config.vaeBaseDim, normalization: .scale(config.vaeScalingFactor))
        try WeightLoading.apply(try checkpoint.loadComponent("transformer"), to: transformer, ignoring: Self.ignoresTransformerKey)
        try WeightLoading.apply(QwenImageVAE.weights(try checkpoint.loadComponent("vae")), to: vae, ignoring: QwenImageVAE.ignoresKey)
        bits = checkpoint.bits
        imageSide = (transformer, vae)
        imageSideUsed = false
    }

    /// Z-Image's pad tokens, which Ming's mapping marks optional and never uses.
    static func ignoresTransformerKey(_ key: String) -> Bool {
        key == "x_pad_token" || key == "cap_pad_token"
    }

    /// The text side, reloaded if it had been released. In low-RAM mode a resident DiT is released
    /// first, so the two are never in memory together.
    public func loadedTextSide() throws -> MingConditionEncoder {
        if let textSide { return textSide }
        if lowRam, imageSideUsed {
            imageSide = nil
            Memory.clearCache()
        }
        try loadTextSide()
        return textSide!
    }

    public func loadedImageSide() throws -> (transformer: S3DiTTransformer, vae: QwenImageVAE) {
        if let imageSide { return imageSide }
        try loadImageSide()
        return imageSide!
    }

    public func isCached(_ prompt: String) -> Bool { promptCache.contains { $0.prompt == prompt } }

    public func encode(_ prompt: String) throws {
        if let index = promptCache.firstIndex(where: { $0.prompt == prompt }) {
            let entry = promptCache.remove(at: index)
            promptCache.append(entry)
            return
        }
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        let features = try encode(promptIds: prompter.tokenIds(prompt))
        promptCache.append((prompt, features))
        if promptCache.count > Self.promptCacheSize { promptCache.removeFirst() }
    }

    /// The caption features of tokenized text: (capFeats [queries, capFeatDim], capFeats2 [tokens, dim]).
    public func encode(promptIds: [Int]) throws -> (MLXArray, MLXArray) {
        let (capFeats, capFeats2) = try loadedTextSide().encode(promptIds: promptIds)
        eval(capFeats, capFeats2)
        return (capFeats, capFeats2)
    }

    public func promptsEncoded() {
        if lowRam, textSide != nil {
            textSide = nil
            Memory.clearCache()
        }
    }

    // MARK: Sampling

    /// The initial noise: [1, 16, H/8, W/8] in float32; Ming keeps its latents plain end to end.
    public static func initialLatents(width: Int, height: Int, seed: Int) -> MLXArray {
        MLXRandom.normal([1, QwenImageVAE.latentChannels, height / 8, width / 8], key: MLXRandom.key(UInt64(seed)))
    }

    /// One transformer pass on [1, 16, h, w] latents: the flow prediction with the same shape. A
    /// `guidance` above 1 runs the official negative condition, both caption streams zeroed.
    public static func predict(
        transformer: S3DiTTransformer, latents: MLXArray, sigma: Float, capFeats: MLXArray, capFeats2: MLXArray, guidance: Float
    ) -> MLXArray {
        let x = latents[0].expandedDimensions(axis: 1)
        let timestep = MLXArray([1 - sigma])
        var out = transformer(latents: x, timestep: timestep, capFeats: capFeats, extraCaption: capFeats2)
        if guidance > 1 {
            let unconditional = transformer(latents: x, timestep: timestep, capFeats: MLXArray.zeros(like: capFeats), extraCaption: MLXArray.zeros(like: capFeats2))
            out = out + guidance * (out - unconditional)
        }
        return out[0..., 0].expandedDimensions(axis: 0)
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        let width = 16 * (request.width / 16)
        let height = 16 * (request.height / 16)
        guard width >= 16, height >= 16 else { throw GenerationError.sizeTooSmall }

        phase(.encoding)
        try encode(request.prompt)
        guard let entry = promptCache.last(where: { $0.prompt == request.prompt }) else { throw GenerationError.cancelled }
        let (capFeats, capFeats2) = entry.features
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let (transformer, vae) = try loadedImageSide()
        let sigmas = StaticShiftSchedule(steps: request.steps, shift: config.sigmaShift).sigmas
        let guidance = Float(request.guidance)
        var latents = Self.initialLatents(width: width, height: height, seed: request.seed)
        for t in 0 ..< request.steps {
            let (sigma, next) = (sigmas[t], sigmas[t + 1])
            // The official schedule ends on a sigma = 0 step whose update is dt = 0: skip its pass.
            if sigma > 0 {
                let velocity = Self.predict(transformer: transformer, latents: latents, sigma: sigma, capFeats: capFeats, capFeats2: capFeats2, guidance: guidance)
                latents = latents + MLXArray(next - sigma) * velocity.asType(.float32)
            }
            eval(latents)
            progress(t + 1, request.steps)
            if isCancelled() { throw GenerationError.cancelled }
        }
        imageSideUsed = true

        phase(.decoding)
        var pixels = try decode(latents: latents, vae: vae, isCancelled: isCancelled)
        if request.flattenAlpha { pixels = Pixels.flattenAlpha(pixels) }
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }

    /// [1, 16, h, w] latents → [H, W, 4] uint8 RGBA pixels, decoded in bf16; in tiles with Save memory on.
    public func decode(latents: MLXArray, vae: QwenImageVAE, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let grid = latents.asType(modelPrecision).transposed(0, 2, 3, 1)
        let decoded = lowRam
            ? try VAETiling.decode(grid, spatialScale: QwenImageVAE.spatialScale, decode: { vae.decode($0) }, isCancelled: isCancelled)
            : vae.decode(grid)
        return Pixels.toPixels(decoded)
    }
}
