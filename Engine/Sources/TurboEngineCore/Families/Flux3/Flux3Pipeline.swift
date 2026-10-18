import Foundation
import MLX
import MLXNN

/// mflux's `Flux3FlowMapScheduler`: `steps + 1` times from 1 to 0, `α·t / (1 + (α − 1)·t)` with
/// α = 6.93, in float32. Each step is one Euler update from a time to the next; the distilled
/// checkpoint (a flow map) is told the next one, the Base runs the same schedule without.
public struct Flux3Schedule: Equatable, Sendable {
    public let timesteps: [Float]

    public init(steps: Int, alpha: Double = 6.93) {
        let t = MLXArray.linspace(Float(1), Float(0), count: steps + 1)
        let shifted = Float(alpha) * t / (1 + Float(alpha - 1) * t)
        timesteps = shifted.asArray(Float.self)
    }
}

/// FLUX 3 Image Klein: Qwen3-VL's language tower for the prompt, the FLUX 3 transformer, FLUX.2's
/// VAE; text to image, and edits from up to ten pictures (`Flux3KleinEdit`), whose tokens go
/// through modes of their own. Reads mflux's checkpoints and BFL's own repository alike (the
/// diffusers folders of `black-forest-labs/flux-3-image-klein`), quantizing an unquantized one
/// on loading when asked, as `mflux-generate-flux3 -q` does.
public final class Flux3Model: PreviewingFamilyModel, LoRAAdaptable {
    public let config: Flux3Config
    public let modelPath: URL
    public let transformer: Flux3Transformer
    public let vae: Flux2VAE
    public private(set) var bits: Int?
    /// Save memory: the text encoder (4.6 GB in 8 bits) goes once the job's prompts are encoded and
    /// is read again for a prompt that was not among them. FLUX.2's decoder would show seams in
    /// tiles, so the decode stays whole.
    public var lowRam = false
    private var residentTextEncoder: Qwen3TextEncoder?
    /// Whether the checkpoint is BFL's (read as mflux reads it), and the bits it is quantized to on
    /// loading: what reading the text encoder again needs.
    private let original: Bool
    private let loadingBits: Int?
    private var prompter: Qwen3Prompter?
    private var promptCache: [String: EncodedPrompt] = [:]
    public let loras = LoRAAdapters(table: .flux3)
    public var adaptedModule: Module? { transformer }

    /// Loads the checkpoint at `modelPath`. A checkpoint without a target time embedding is the
    /// undistilled Base, whatever `config` says (`flowMap: false` reads a distilled one as a Base,
    /// for the parity check). `quantize` (4, 6, 8) applies to an unquantized checkpoint, every
    /// linear and embedding in groups of 64 as mflux quantizes them.
    public init(modelPath: URL, config: Flux3Config, quantize: Int? = nil, flowMap: Bool? = nil, loadTokenizer: Bool = true) throws {
        self.modelPath = modelPath
        var checkpoint = Checkpoint(root: modelPath)
        var transformerWeights = try checkpoint.loadComponent("transformer")
        var config = config
        config.transformer.flowMap = flowMap ?? transformerWeights.keys.contains { $0.hasPrefix("target_timestep_embedder.") }
        if !config.transformer.flowMap { transformerWeights = transformerWeights.filter { !$0.key.hasPrefix("target_timestep_embedder.") } }
        config.defaultSteps = config.isBase ? 50 : 16
        self.config = config
        transformer = Flux3Transformer(config: config.transformer)
        vae = Flux2VAE()

        // BFL's repository has no mflux metadata: its tensors are read as mflux reads them, every
        // one in bfloat16, under the module's names.
        let original = checkpoint.mfluxVersion == nil
        let loadingBits = checkpoint.bits == nil ? quantize : nil
        self.original = original
        self.loadingBits = loadingBits
        try WeightLoading.apply(Self.prepared(Self.transformerKeys(transformerWeights), original: original), to: transformer)
        bits = checkpoint.bits ?? loadingBits
        var vaeWeights = try checkpoint.loadComponent("vae")
        if original { vaeWeights = Self.originalVAEKeys(vaeWeights) }
        try WeightLoading.apply(Self.prepared(vaeWeights, original: original), to: vae, ignoring: { $0.hasSuffix("num_batches_tracked") })
        if let loadingBits {
            for module in [transformer, vae] as [Module] { Self.quantize(module, bits: loadingBits) }
        }
        residentTextEncoder = try Self.loadTextEncoder(modelPath: modelPath, config: config, original: original, quantize: loadingBits)
        if loadTokenizer {
            prompter = try Qwen3Prompter(modelPath: modelPath, folder: "text_encoder", maxLength: config.maxSequenceLength,
                                         enableThinking: true)
        }
    }

    /// Every linear and embedding in groups of 64, as mflux quantizes on loading.
    static func quantize(_ module: Module, bits: Int) {
        MLXNN.quantize(model: module, groupSize: 64, bits: bits) { _, layer in layer is Linear || layer is Embedding }
    }

    /// BFL's tensors in bfloat16, as mflux casts them; an mflux checkpoint's as saved.
    static func prepared(_ weights: [String: MLXArray], original: Bool) -> [String: MLXArray] {
        guard original else { return weights }
        return weights.mapValues { $0.dtype.isFloatingPoint && $0.dtype != .bfloat16 ? $0.asType(.bfloat16) : $0 }
    }

    /// The text encoder from the checkpoint (lazily: its weights are read when it first runs).
    static func loadTextEncoder(modelPath: URL, config: Flux3Config, original: Bool, quantize: Int?) throws -> Qwen3TextEncoder {
        let encoder = Qwen3TextEncoder(config: config.textEncoder)
        var checkpoint = Checkpoint(root: modelPath)
        let weights = textEncoderKeys(try checkpoint.loadComponent("text_encoder"), layers: config.textEncoder.numHiddenLayers)
        try WeightLoading.apply(prepared(weights, original: original), to: encoder)
        if let quantize { Self.quantize(encoder, bits: quantize) }
        return encoder
    }

    /// The text encoder, read again if Save memory had released it.
    public func textEncoder() throws -> Qwen3TextEncoder {
        if let residentTextEncoder { return residentTextEncoder }
        let encoder = try Self.loadTextEncoder(modelPath: modelPath, config: config, original: original, quantize: loadingBits)
        residentTextEncoder = encoder
        return encoder
    }

    /// The transformer's keys: diffusers' `attn.to_out.0` is the module's `attn.to_out`.
    static func transformerKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: weights.map { ($0.key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out."), $0.value) })
    }

    /// The text encoder's keys: Qwen3-VL's language tower (`model.language_model.*`) without the
    /// vision tower, the head and the layers past the last one read; mflux's own keys unchanged.
    static func textEncoderKeys(_ weights: [String: MLXArray], layers: Int) -> [String: MLXArray] {
        var kept: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("model.language_model.") {
                name.removeFirst("model.language_model.".count)
            } else if name.hasPrefix("model.") || name.hasPrefix("lm_head.") || name.hasPrefix("visual.") {
                continue
            }
            if name.hasPrefix("rotary_emb.") { continue }
            if name.hasPrefix("layers."), let index = Int(name.split(separator: ".")[1]), index >= layers { continue }
            kept[name] = value
        }
        return kept
    }

    /// The VAE of BFL's repository (diffusers' layout) as mflux maps it: `to_out.0` is `to_out`, and
    /// convolution weights go from [O, I, H, W] to MLX's [O, H, W, I].
    static func originalVAEKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in weights {
            let name = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            mapped[name] = value.ndim == 4 && name.hasSuffix(".weight") ? value.transposed(0, 2, 3, 1) : value
        }
        return mapped
    }

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil }

    public func encode(_ prompt: String) throws {
        _ = try encodePrompt(prompt)
    }

    public func promptsEncoded() {
        if lowRam, residentTextEncoder != nil {
            residentTextEncoder = nil
            Memory.clearCache()
        }
    }

    public func dropPromptCache() { promptCache.removeAll() }

    /// Encodes a prompt (once; later calls read the cache).
    @discardableResult
    public func encodePrompt(_ prompt: String) throws -> EncodedPrompt {
        if let cached = promptCache[prompt] { return cached }
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        let (inputIds, attentionMask) = try prompter.tokenize(prompt, multipleOf: config.padMultiple)
        let encoded = try encode(inputIds: inputIds, attentionMask: attentionMask)
        promptCache[prompt] = encoded
        return encoded
    }

    /// The encoder on already tokenized input (what the fixture check feeds it): the eight hidden
    /// states side by side for every token, padding included, in bfloat16.
    public func encode(inputIds: MLXArray, attentionMask: MLXArray, computeType: DType? = nil) throws -> EncodedPrompt {
        let embeds = try textEncoder().promptEmbeds(
            inputIds: inputIds, attentionMask: attentionMask, positions: Self.positions(attentionMask: attentionMask),
            layers: config.textEncoderOutLayers, computeType: computeType
        ).asType(computeType ?? .bfloat16)
        let ids = Self.textIds(count: embeds.shape[1])
        eval(embeds, ids)
        return EncodedPrompt(embeds: embeds, ids: ids)
    }

    /// `Flux3PromptEncoder.positions`: `cumsum(mask) − 1`, the padding at position 1.
    public static func positions(attentionMask: MLXArray) -> MLXArray {
        let mask = attentionMask.asType(.int32)
        return MLX.where(mask .== 0, MLXArray(Int32(1)), cumsum(mask, axis: -1) - 1)
    }

    /// [count, 4] rows of (0, 0, 0, token index), float32.
    public static func textIds(count: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: count * 4)
        for index in 0 ..< count { values[index * 4 + 3] = Float(index) }
        return MLXArray(values, [count, 4])
    }

    // MARK: Sampling

    /// The initial noise for a size and seed, [1, h·w, 128]: drawn in float32, rounded to
    /// bfloat16 as the reference rounds it, kept in float32 (the loop's state).
    public static func initialNoise(width: Int, height: Int, seed: Int) -> MLXArray {
        MLXRandom.normal([1, (height / 16) * (width / 16), 128], key: MLXRandom.key(UInt64(seed)))
            .asType(.bfloat16).asType(.float32)
    }

    /// [h·w, 4] rows of (0, row/2, column/2, layer), row-major like the latents: layer 0 for the
    /// image, −1, −2… for the pictures it is edited from.
    public static func imageIds(height: Int, width: Int, layer: Int = 0) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let row = (y * width + x) * 4
                values[row + 1] = Float(y) * 0.5
                values[row + 2] = Float(x) * 0.5
                values[row + 3] = Float(layer)
            }
        }
        return MLXArray(values, [height * width, 4])
    }

    /// The prepared pictures as the tokens of the reference modes [1, R, 128] (bfloat16) and their ids.
    public func referenceTokens(_ pictures: [Flux3References.Picture]) -> (tokens: MLXArray, ids: MLXArray) {
        var tokens: [MLXArray] = []
        var ids: [MLXArray] = []
        for (index, picture) in pictures.enumerated() {
            let encoded = vae.encodePacked(Flux3References.pixels(picture))
            tokens.append(encoded.tokens.asType(.bfloat16))
            ids.append(Self.imageIds(height: encoded.height, width: encoded.width, layer: -(1 + index)))
        }
        let result = (tokens: concatenated(tokens, axis: 1), ids: concatenated(ids, axis: 0))
        eval(result.tokens, result.ids)
        return result
    }

    /// One Euler update to the next time: `latents + Δt · velocity`, Δt rounded to the velocity's
    /// dtype as the reference's Python float is, the sum in float32.
    public static func step(latents: MLXArray, velocity: MLXArray, schedule: Flux3Schedule, index: Int) -> MLXArray {
        let dt = schedule.timesteps[index + 1] - schedule.timesteps[index]
        return latents + dt * velocity
    }

    /// The velocity at step `index`: the transformer's, or classifier-free guidance's
    /// `u + g·(c − u)` in float32 against the empty prompt for the Base.
    public func velocity(
        latents: MLXArray, prompt: EncodedPrompt, rotary: Flux3Rotary, unconditional: (prompt: EncodedPrompt, rotary: Flux3Rotary)?,
        guidance: Double, references: MLXArray?, schedule: Flux3Schedule, index: Int
    ) -> MLXArray {
        let timestep = schedule.timesteps[index]
        let target = transformer.flowMap ? schedule.timesteps[index + 1] : nil
        let conditional = transformer(latents: latents, prompt: prompt.embeds, references: references, timestep: timestep,
                                      targetTimestep: target, rotary: rotary)
        guard let unconditional else { return conditional }
        let u = transformer(latents: latents, prompt: unconditional.prompt.embeds, references: references, timestep: timestep,
                            targetTimestep: target, rotary: unconditional.rotary).asType(.float32)
        return u + Float(guidance) * (conditional.asType(.float32) - u)
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        preview: (Preview) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        try generate(
            prompt: request.prompt, seed: request.seed, width: request.width, height: request.height, steps: request.steps,
            guidance: request.guidance, referencePaths: request.imagePaths,
            phase: phase, progress: progress, preview: preview, isCancelled: isCancelled
        )
    }

    /// Runs the whole pipeline for a prompt. The distilled checkpoint ignores `guidance`; the Base
    /// runs classifier-free guidance at it against the empty prompt. With `referencePaths` (at most
    /// ten), the image is edited from those pictures, prepared as `Flux3ReferenceImages` prepares
    /// them; a single picture asked for at its own size is made on its padded canvas and cropped
    /// back, as mflux does when no size is given.
    public func generate(
        prompt: String, seed: Int, width: Int, height: Int, steps: Int, guidance: Double = 3.5,
        referencePaths: [String] = [],
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        preview: (Preview) -> Void = { _ in },
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        phase(.encoding)
        let encoded = try encodePrompt(prompt)
        let unconditionalPrompt = config.isBase ? try encodePrompt("") : nil
        // The size asked for is compared with a single picture's own before it is cut to multiples
        // of 16: that is the case made on the padded canvas.
        var (width, height) = (width, height)
        var crop: Flux3References.Box?
        var reference: (tokens: MLXArray, ids: MLXArray)?
        if !referencePaths.isEmpty {
            let prepared = try Flux3References.prepare(try referencePaths.map(Flux3References.load))
            let canvas = Flux3References.resolveCanvas(prepared.pictures, box: prepared.box, width: width, height: height)
            (width, height, crop) = (canvas.width, canvas.height, canvas.crop)
            reference = referenceTokens(prepared.pictures)
        }
        (width, height) = (16 * (width / 16), 16 * (height / 16))
        guard width >= 16, height >= 16 else { throw GenerationError.sizeTooSmall }
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let (latentHeight, latentWidth) = (height / 16, width / 16)
        let ids = Self.imageIds(height: latentHeight, width: latentWidth)
        let rotary = transformer.rotary(imageIds: ids, referenceIds: reference?.ids, textIds: encoded.ids)
        let unconditional = unconditionalPrompt.map {
            (prompt: $0, rotary: transformer.rotary(imageIds: ids, referenceIds: reference?.ids, textIds: $0.ids))
        }
        let schedule = Flux3Schedule(steps: steps)
        var latents = Self.initialNoise(width: width, height: height, seed: seed)
        for t in 0 ..< steps {
            let velocity = velocity(latents: latents, prompt: encoded, rotary: rotary, unconditional: unconditional,
                                    guidance: guidance, references: reference?.tokens, schedule: schedule, index: t)
            let glimpse = LatentPreview.shows(step: t + 1, of: steps)
                ? LatentPreview.predicted(latents: latents, noise: velocity, sigma: schedule.timesteps[t]) : nil
            latents = Self.step(latents: latents, velocity: velocity, schedule: schedule, index: t)
            eval(latents)
            progress(t + 1, steps)
            if let glimpse {
                preview(Preview(pixels: previewPixels(latents: glimpse, latentHeight: latentHeight, latentWidth: latentWidth), step: t + 1))
            }
            if isCancelled() { throw GenerationError.cancelled }
        }

        phase(.decoding)
        var pixels = decode(latents: latents, latentHeight: latentHeight, latentWidth: latentWidth)
        if let crop { pixels = pixels[crop.top ..< crop.bottom, crop.left ..< crop.right, 0...] }
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }

    /// A small image of what packed latents [1, h·w, 128] (a step's prediction) hold.
    func previewPixels(latents: MLXArray, latentHeight: Int, latentWidth: Int) -> MLXArray {
        let grid = latents.asType(.bfloat16).reshaped([1, latentHeight, latentWidth, latents.shape[latents.ndim - 1]])
        let factor = LatentPreview.factor(height: latentHeight, width: latentWidth, scale: 16)
        return Pixels.toPixels(vae.decodePacked(LatentPreview.pooled(grid, factor: factor)))
    }

    /// Packed latents → [H, W, 3] uint8 pixels, as `ImageUtil.to_image` converts them. Off the
    /// reference's path by design: mflux decodes the loop's float32 state, which keeps FLUX.2's
    /// whole decoder (its residual stream) in float32; rounded to bfloat16 first, as Klein's are,
    /// the decode peaks at 31 GB instead of 43–49 at 2048 × 2048 (22 instead of 34 at 1536) and is a
    /// quarter faster, the image 47–48 dB from the float32 decode (0.1–0.3 dB further from mflux's).
    public func decode(latents: MLXArray, latentHeight: Int, latentWidth: Int) -> MLXArray {
        let grid = latents.asType(.bfloat16).reshaped([1, latentHeight, latentWidth, latents.shape[latents.ndim - 1]])
        return Pixels.toPixels(vae.decodePacked(grid))
    }
}
