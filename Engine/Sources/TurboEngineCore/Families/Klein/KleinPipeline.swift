import Foundation
import MLX
import MLXNN

/// mflux's `FlowMatchEulerDiscreteScheduler` for a text-to-image run: `steps + 1` sigmas ending
/// in 0 and one timestep per step in [0, 1000].
public struct FlowMatchSchedule: Equatable, Sendable {
    public let sigmas: [Float]
    public let timesteps: [Float]

    public init(steps: Int) {
        let trainSteps = 1000.0
        let shiftTerminal = 0.02
        guard steps > 1 else {
            sigmas = [1, 0]
            timesteps = [Float(trainSteps)]
            return
        }
        let sigmaMin = 1 / trainSteps
        let linear = (0 ..< steps).map { i in
            trainSteps - Double(i) * (1 - sigmaMin) * trainSteps / Double(steps - 1)
        }
        let shifted = linear.map { t -> Double in
            let sigma = t / trainSteps
            return exp(1.0) / (exp(1.0) + (1 / sigma - 1))
        }
        let oneMinus = shifted.map { 1 - $0 }
        let scale = oneMinus[oneMinus.count - 1] / (1 - shiftTerminal)
        let stretched = oneMinus.map { 1 - $0 / scale }
        sigmas = stretched.map { Float($0) } + [0]
        timesteps = stretched.map { Float($0 * trainSteps) }
    }

    /// The schedule FLUX.2 actually runs: its model configs set `requires_sigma_shift`, so mflux's
    /// `Config` replaces the one above with `set_image_seq_len`: `steps` sigmas from 1 down to
    /// 1/steps, shifted by an empirical mu of the image's token count and the step count, then 0.
    public init(steps: Int, imageSeqLen: Int) {
        let expMu = Float(exp(Self.empiricalMu(imageSeqLen: imageSeqLen, steps: steps)))
        let shifted = (0 ..< steps).map { i -> Float in
            // mx.linspace(1, 1/steps, steps) in float32.
            let t = steps > 1 ? 1 + Float(i) * (1 / Float(steps) - 1) / Float(steps - 1) : 1
            return expMu / (expMu + (1 / t - 1))
        }
        sigmas = shifted + [0]
        timesteps = shifted.map { $0 * 1000 }
    }

    /// `_compute_empirical_mu`: a fit over the sequence length, interpolated in the step count.
    static func empiricalMu(imageSeqLen: Int, steps: Int) -> Double {
        let (a1, b1) = (8.73809524e-05, 1.89833333)
        let (a2, b2) = (0.00016927, 0.45666666)
        let length = Double(imageSeqLen)
        if imageSeqLen > 4300 { return a2 * length + b2 }
        let m200 = a2 * length + b2
        let m10 = a1 * length + b1
        let a = (m200 - m10) / 190
        let b = m200 - 200 * a
        return a * Double(steps) + b
    }
}

/// What the text encoder produces for a prompt: the embeddings and the ids the transformer's
/// rotary embedding reads for them.
public struct EncodedPrompt {
    public let embeds: MLXArray
    public let ids: MLXArray
}

public struct GeneratedImage {
    /// Pixels as [H, W, 3] (or [H, W, 4]) uint8.
    public let pixels: MLXArray
    public var width: Int { pixels.shape[1] }
    public var height: Int { pixels.shape[0] }
}

/// Reported while an image is generated, as `turbo_worker.py` reports it.
public enum GenerationPhase: String {
    case loading, encoding, denoising, decoding, saving
}

public enum GenerationError: LocalizedError {
    case cancelled
    case sizeTooSmall

    public var errorDescription: String? {
        switch self {
        case .cancelled: "Cancelled"
        case .sizeTooSmall: "The image must be at least 16 × 16 pixels."
        }
    }
}

/// FLUX.2 Klein: the three modules, the prompt cache, and the sampling loop.
public final class KleinModel: FamilyModel {
    /// The negative prompt mflux encodes for a base checkpoint's classifier-free guidance.
    static let negativePrompt = " "

    public let config: KleinConfig
    public let modelPath: URL
    public let textEncoder: Qwen3TextEncoder
    public let transformer: Flux2Transformer
    public let vae: Flux2VAE
    public private(set) var bits: Int?
    /// FLUX.2's decoder would show seams in tiles, so Save memory changes nothing here.
    public var lowRam = false
    private var prompter: Qwen3Prompter?
    private var promptCache: [String: EncodedPrompt] = [:]

    /// Loads the checkpoint at `modelPath` (mflux format). Weights stay lazy until first use.
    public init(modelPath: URL, config: KleinConfig, loadTokenizer: Bool = true) throws {
        self.config = config
        self.modelPath = modelPath
        textEncoder = Qwen3TextEncoder(config: config.textEncoder)
        transformer = Flux2Transformer(config: config.transformer)
        vae = Flux2VAE()

        var checkpoint = Checkpoint(root: modelPath)
        try WeightLoading.apply(try checkpoint.loadComponent("text_encoder"), to: textEncoder, ignoring: Qwen3TextEncoder.ignoresKey)
        try WeightLoading.apply(try checkpoint.loadComponent("transformer"), to: transformer)
        try WeightLoading.apply(try checkpoint.loadComponent("vae"), to: vae, ignoring: Flux2VAE.ignoresKey)
        bits = checkpoint.bits
        if loadTokenizer {
            prompter = try Qwen3Prompter(modelPath: modelPath, maxLength: config.maxSequenceLength, enableThinking: false)
        }
    }

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil }

    public func encode(_ prompt: String) throws {
        _ = try encodePrompt(prompt)
    }

    public func promptsEncoded() {}

    /// Encodes a prompt (once; later calls read the cache).
    @discardableResult
    public func encodePrompt(_ prompt: String) throws -> EncodedPrompt {
        if let cached = promptCache[prompt] { return cached }
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        let (inputIds, attentionMask) = try prompter.tokenize(prompt)
        let encoded = encode(inputIds: inputIds, attentionMask: attentionMask)
        promptCache[prompt] = encoded
        return encoded
    }

    /// The encoder on already tokenized input (what the fixture check feeds it).
    public func encode(inputIds: MLXArray, attentionMask: MLXArray) -> EncodedPrompt {
        let embeds = textEncoder.promptEmbeds(inputIds: inputIds, attentionMask: attentionMask, layers: config.textEncoderOutLayers)
        let ids = Self.textIds(count: embeds.shape[1])
        eval(embeds, ids)
        return EncodedPrompt(embeds: embeds, ids: ids)
    }

    public func dropPromptCache() { promptCache.removeAll() }

    // MARK: Sampling

    /// The initial noise for a size and seed, packed as [1, S, 128], with its latent grid.
    public static func initialLatents(width: Int, height: Int, seed: Int) -> (latents: MLXArray, ids: MLXArray, latentHeight: Int, latentWidth: Int) {
        let latentHeight = height / 16
        let latentWidth = width / 16
        let noise = MLXRandom.normal([1, 128, latentHeight, latentWidth], key: MLXRandom.key(UInt64(seed)))
            .asType(modelPrecision)
        let packed = noise.reshaped([1, 128, latentHeight * latentWidth]).transposed(0, 2, 1)
        return (packed, imageIds(height: latentHeight, width: latentWidth), latentHeight, latentWidth)
    }

    /// [1, h·w, 4] rows of (0, y, x, 0), row-major like the packed latents.
    static func imageIds(height: Int, width: Int) -> MLXArray {
        var values: [Int32] = []
        values.reserveCapacity(height * width * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                values.append(contentsOf: [0, Int32(y), Int32(x), 0])
            }
        }
        return MLXArray(values, [1, height * width, 4])
    }

    /// [1, count, 4] rows of (0, 0, 0, token index).
    static func textIds(count: Int) -> MLXArray {
        var values: [Int32] = []
        values.reserveCapacity(count * 4)
        for index in 0 ..< count { values.append(contentsOf: [0, 0, 0, Int32(index)]) }
        return MLXArray(values, [1, count, 4])
    }

    /// One Euler step of the flow: `latents + (σ[t+1] − σ[t]) · noise`, in the latents' dtype.
    public static func step(latents: MLXArray, noise: MLXArray, schedule: FlowMatchSchedule, index: Int) -> MLXArray {
        let dt = MLXArray(schedule.sigmas[index + 1] - schedule.sigmas[index]).asType(latents.dtype)
        return latents + dt * noise.asType(latents.dtype)
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        // Distilled checkpoints only work at guidance 1; base checkpoints run real CFG.
        let guidance = config.isBase ? request.guidance : 1
        return try generate(
            prompt: request.prompt, seed: request.seed, width: request.width, height: request.height,
            steps: request.steps, guidance: guidance, phase: phase, progress: progress, isCancelled: isCancelled
        )
    }

    /// Runs the whole pipeline for a prompt. `progress` gets each finished step; `isCancelled`
    /// is consulted between steps. A `guidance` above 1 runs mflux's classifier-free guidance
    /// against an encoded space, as its base checkpoints do.
    public func generate(
        prompt: String, seed: Int, width: Int, height: Int, steps: Int, guidance: Double = 1,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        let width = 16 * (width / 16)
        let height = 16 * (height / 16)
        guard width >= 16, height >= 16 else { throw GenerationError.sizeTooSmall }

        phase(.encoding)
        let encoded = try encodePrompt(prompt)
        let negative = guidance > 1 ? try encodePrompt(Self.negativePrompt) : nil
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let initial = Self.initialLatents(width: width, height: height, seed: seed)
        let schedule = FlowMatchSchedule(steps: steps, imageSeqLen: initial.latentHeight * initial.latentWidth)
        var latents = initial.latents
        for t in 0 ..< steps {
            var noise = transformer(latents: latents, prompt: encoded.embeds, timestep: schedule.timesteps[t], imageIds: initial.ids, textIds: encoded.ids)
            if let negative {
                let negativeNoise = transformer(latents: latents, prompt: negative.embeds, timestep: schedule.timesteps[t], imageIds: initial.ids, textIds: negative.ids)
                noise = negativeNoise + Float(guidance) * (noise - negativeNoise)
            }
            latents = Self.step(latents: latents, noise: noise, schedule: schedule, index: t)
            eval(latents)
            progress(t + 1, steps)
            if isCancelled() { throw GenerationError.cancelled }
        }

        phase(.decoding)
        let pixels = decode(latents: latents, latentHeight: initial.latentHeight, latentWidth: initial.latentWidth)
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }

    /// Packed latents → [H, W, 3] uint8 pixels, as `ImageUtil.to_image` converts them.
    public func decode(latents: MLXArray, latentHeight: Int, latentWidth: Int) -> MLXArray {
        let grid = latents.reshaped([1, latentHeight, latentWidth, latents.shape[latents.ndim - 1]])
        let decoded = vae.decodePacked(grid)
        return Pixels.toPixels(decoded)
    }
}
