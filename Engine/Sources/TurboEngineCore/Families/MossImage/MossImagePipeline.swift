import Foundation
import MLX
import MLXNN

/// MOSS-Image (OpenMOSS, Apache 2.0): Qwen3.5-9B's last hidden state for the prompt, a 5B single-stream
/// DiT, FLUX.2's VAE with BFL's small decoder. Classifier-free guidance runs its negative branch with
/// block 22 skipped, as `MOSSImagePipeline` does by default. Reads the diffusers folders of
/// OpenMOSS's repositories (and their byte-identical `mlx-community/MOSS-Image-*-bf16` mirrors),
/// quantizing the transformer and the text encoder on loading when asked.
public final class MossImageModel: PreviewingFamilyModel {
    public let config: MossImageConfig
    public let modelPath: URL
    public let vae: Flux2VAE
    public private(set) var bits: Int?
    /// Save memory: the text encoder (17 GB in bf16, 8 in 8 bits) goes once the job's prompts are
    /// encoded, and for a prompt that was not among them the transformer (5 GB in 8 bits) goes before
    /// it is read again, so the two are never in memory together. The decode stays whole.
    public var lowRam = false
    private var residentTransformer: MossImageTransformer?
    private var residentTextEncoder: Qwen35TextEncoder?
    private let loadingBits: Int?
    private var prompter: Qwen3Prompter?
    private var promptCache: [String: MLXArray] = [:]

    /// Loads the checkpoint at `modelPath`. `quantize` (4, 6, 8) applies to the transformer's and the
    /// text encoder's linears and embeddings, in groups of 64; the weights otherwise keep the dtype
    /// they were saved in (bf16, or float32 in a fixture), the VAE in bf16 as the other families'.
    public init(modelPath: URL, quantize: Int? = nil, loadTokenizer: Bool = true) throws {
        self.modelPath = modelPath
        config = MossImageConfig(modelPath: modelPath)
        var checkpoint = Checkpoint(root: modelPath)
        vae = Flux2VAE(decoderChannels: config.decoderChannels)
        let vaeWeights = Flux3Model.originalVAEKeys(try checkpoint.loadComponent("vae"))
            .mapValues { $0.dtype.isFloatingPoint ? $0.asType(modelPrecision) : $0 }
        try WeightLoading.apply(vaeWeights, to: vae, ignoring: { $0.hasSuffix("num_batches_tracked") })
        loadingBits = quantize
        bits = quantize
        residentTransformer = try Self.loadTransformer(modelPath: modelPath, config: config, quantize: quantize)
        residentTextEncoder = try Self.loadTextEncoder(modelPath: modelPath, config: config, quantize: quantize)
        if loadTokenizer {
            prompter = try Qwen3Prompter(modelPath: modelPath, folder: "tokenizer", maxLength: config.maxSequenceLength,
                                         enableThinking: true, template: .qwen35)
        }
    }

    /// Every linear and embedding in groups of 64 (the ones whose input width allows it).
    static func quantize(_ module: Module, bits: Int) {
        MLXNN.quantize(model: module, groupSize: 64, bits: bits) { _, layer in
            if let linear = layer as? Linear { return linear.weight.dim(1) % 64 == 0 }
            return layer is Embedding
        }
    }

    /// The transformer from the checkpoint (lazily: its weights are read when it first runs).
    static func loadTransformer(modelPath: URL, config: MossImageConfig, quantize: Int?) throws -> MossImageTransformer {
        let transformer = MossImageTransformer(config: config.transformer)
        var checkpoint = Checkpoint(root: modelPath)
        try WeightLoading.apply(MossImageTransformer.keys(try checkpoint.loadComponent("transformer")), to: transformer)
        if let quantize { Self.quantize(transformer, bits: quantize) }
        return transformer
    }

    /// The transformer, read again if Save memory had released it.
    public var transformer: MossImageTransformer {
        get throws {
            if let residentTransformer { return residentTransformer }
            let transformer = try Self.loadTransformer(modelPath: modelPath, config: config, quantize: loadingBits)
            residentTransformer = transformer
            return transformer
        }
    }

    /// The text encoder from the checkpoint (lazily: its weights are read when it first runs).
    static func loadTextEncoder(modelPath: URL, config: MossImageConfig, quantize: Int?) throws -> Qwen35TextEncoder {
        let encoder = Qwen35TextEncoder(config: config.textEncoder)
        var checkpoint = Checkpoint(root: modelPath)
        try WeightLoading.apply(Qwen35TextEncoder.keys(try checkpoint.loadComponent("text_encoder")), to: encoder)
        if let quantize { Self.quantize(encoder, bits: quantize) }
        return encoder
    }

    /// The text encoder, read again if Save memory had released it (after the transformer, with Save memory).
    public func textEncoder() throws -> Qwen35TextEncoder {
        if let residentTextEncoder { return residentTextEncoder }
        if lowRam, residentTransformer != nil {
            residentTransformer = nil
            Memory.clearCache()
        }
        let encoder = try Self.loadTextEncoder(modelPath: modelPath, config: config, quantize: loadingBits)
        residentTextEncoder = encoder
        return encoder
    }

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil }

    /// Encodes a prompt, and guidance's empty negative prompt with it, while the encoder is resident.
    public func encode(_ prompt: String) throws {
        _ = try encodePrompt(prompt)
        _ = try encodePrompt("")
    }

    public func promptsEncoded() {
        if lowRam, residentTextEncoder != nil {
            residentTextEncoder = nil
            Memory.clearCache()
        }
    }

    /// `encode_prompt` for one prompt (once; later calls read the cache): the chat template with
    /// thinking on, at most 4096 tokens, the last hidden state of every token [1, S, 4096].
    @discardableResult
    public func encodePrompt(_ prompt: String) throws -> MLXArray {
        if let cached = promptCache[prompt] { return cached }
        let ids = try tokenIds(prompt)
        let encoded = try encode(ids: MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        promptCache[prompt] = encoded
        return encoded
    }

    /// The prompt's token ids as `encode_prompt` makes them.
    public func tokenIds(_ prompt: String) throws -> [Int] {
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        return try prompter.tokenIds(prompt)
    }

    /// The encoder on token ids [1, S] (what the fixture check feeds it).
    public func encode(ids: MLXArray, computeType: DType? = nil) throws -> MLXArray {
        let hidden = try textEncoder().lastHiddenState(ids, computeType: computeType)
        eval(hidden)
        return hidden
    }

    // MARK: Sampling

    /// The initial noise for a size and seed, [1, h, w, 128] channels last, drawn in float32 (the
    /// reference samples it in float32 whatever the weights).
    public static func initialNoise(width: Int, height: Int, seed: Int, channels: Int = 128) -> MLXArray {
        MLXRandom.normal([1, height / 16, width / 16, channels], key: MLXRandom.key(UInt64(seed)))
    }

    /// The guided velocity at a timestep in float32: the prompt's, and with guidance (any scale but 1)
    /// `u + g·(c − u)` against the negative prompt run with `skipLayers` left out.
    public func velocity(
        latents: MLXArray, timestep: Float, prompt: MLXArray, negative: MLXArray?, guidance: Float,
        skipLayers: Set<Int>
    ) throws -> MLXArray {
        let transformer = try self.transformer
        let conditional = transformer(latents, timestep: timestep, context: prompt).asType(.float32)
        guard let negative, guidance != 1 else { return conditional }
        let unconditional = transformer(latents, timestep: timestep, context: negative, skipLayers: skipLayers).asType(.float32)
        return unconditional + guidance * (conditional - unconditional)
    }

    /// The denoising loop from `noise` [1, h, w, 128] (float32): one Euler step per scheduled sigma.
    public func denoise(
        noise: MLXArray, prompt: MLXArray, negative: MLXArray?, steps: Int, guidance: Float,
        progress: (Int, Int) -> Void = { _, _ in }, preview: (Int, MLXArray) -> Void = { _, _ in },
        isCancelled: () -> Bool = { false }
    ) throws -> MLXArray {
        let schedule = MossImageSchedule(steps: steps, config: config)
        var latents = noise.asType(.float32)
        let total = schedule.steps
        for index in 0 ..< total {
            let velocity = try velocity(latents: latents, timestep: schedule.timesteps[index], prompt: prompt, negative: negative,
                                    guidance: guidance, skipLayers: config.unconditionalSkipLayers)
            let glimpse = LatentPreview.shows(step: index + 1, of: total)
                ? LatentPreview.predicted(latents: latents, noise: velocity, sigma: schedule.sigmas[index]) : nil
            latents = latents + (schedule.sigmas[index + 1] - schedule.sigmas[index]) * velocity
            eval(latents)
            progress(index + 1, total)
            if let glimpse { preview(index + 1, glimpse) }
            if isCancelled() { throw GenerationError.cancelled }
        }
        return latents
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        preview: (Preview) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        phase(.encoding)
        let prompt = try encodePrompt(request.prompt)
        let negative = request.guidance != 1 ? try encodePrompt(request.negativePrompt ?? "") : nil
        let (width, height) = (16 * (request.width / 16), 16 * (request.height / 16))
        guard width >= 16, height >= 16 else { throw GenerationError.sizeTooSmall }
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let noise = Self.initialNoise(width: width, height: height, seed: request.seed, channels: config.transformer.latentChannels)
        let latents = try denoise(
            noise: noise, prompt: prompt, negative: negative, steps: request.steps, guidance: Float(request.guidance),
            progress: progress,
            preview: { step, glimpse in preview(Preview(pixels: self.previewPixels(glimpse), step: step)) },
            isCancelled: isCancelled
        )

        phase(.decoding)
        let pixels = decode(latents)
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }

    /// A small image of what latents [1, h, w, 128] (a step's prediction) hold.
    func previewPixels(_ latents: MLXArray) -> MLXArray {
        let factor = LatentPreview.factor(height: latents.dim(1), width: latents.dim(2), scale: 16)
        return Pixels.toPixels(vae.decodePacked(LatentPreview.pooled(latents.asType(modelPrecision), factor: factor)))
    }

    /// Latents [1, h, w, 128] → [H, W, 3] uint8: the running statistics undone, the 2 × 2 patches
    /// unshuffled, decoded, clamped and rounded as `VaeImageProcessor.postprocess` does. The
    /// reference decodes in float32; here in bf16, as the other FLUX.2-VAE families.
    public func decode(_ latents: MLXArray) -> MLXArray {
        Pixels.toPixels(vae.decodePacked(latents.asType(modelPrecision)))
    }
}
