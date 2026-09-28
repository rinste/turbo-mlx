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
    case unreadableImage(String)
    case referenceTooSmall

    public var errorDescription: String? {
        switch self {
        case .cancelled: "Cancelled"
        case .sizeTooSmall: "The image must be at least 16 × 16 pixels."
        case .unreadableImage(let path): "Couldn’t read the reference image \((path as NSString).lastPathComponent)."
        case .referenceTooSmall: "The reference image must be at least 16 × 16 pixels."
        }
    }
}

/// A picture an image is edited from, as `_Flux2KleinEditHelpers.prepare_reference_image`
/// prepares it: in its own proportions whatever the image's size, scaled down to at most about a
/// megapixel, its sides then cut to a multiple of 16 from the middle.
public enum KleinReference {
    /// The most pixels a reference is encoded at, as the diffusers pipeline caps it.
    static let maxArea = 1024 * 1024

    /// A `width` × `height` picture once scaled to fit `maxArea` (rounded half to even, as Python
    /// rounds), and the size it is encoded at: that one cut down to multiples of 16.
    public static func sizes(width: Int, height: Int) -> (scaled: (width: Int, height: Int), encoded: (width: Int, height: Int)) {
        var scaled = (width: width, height: height)
        if width * height > maxArea {
            let scale = (Double(maxArea) / Double(width * height)).squareRoot()
            scaled = (roundHalfEven(Double(width) * scale), roundHalfEven(Double(height) * scale))
        }
        return (scaled, (scaled.width - scaled.width % 16, scaled.height - scaled.height % 16))
    }

    /// The picture at `path` as the encoder takes it: [1, H, W, 3] in [-1, 1], in float32 as mflux
    /// hands it over.
    public static func pixels(path: String) throws -> MLXArray {
        guard let image = ImagePixels.load(path) else { throw GenerationError.unreadableImage(path) }
        let (scaled, encoded) = sizes(width: image.width, height: image.height)
        guard encoded.width > 0, encoded.height > 0 else { throw GenerationError.referenceTooSmall }
        var rgba = ImagePixels.rgbaBytes(image)
        if scaled.width != image.width || scaled.height != image.height {
            rgba = ImagePixels.resize(rgba, width: image.width, height: image.height, toWidth: scaled.width, toHeight: scaled.height)
        }
        let rgb = ImagePixels.crop(rgba, rgbaWidth: scaled.width, left: (scaled.width - encoded.width) / 2,
                                   top: (scaled.height - encoded.height) / 2, width: encoded.width, height: encoded.height)
        let bytes = MLXArray(rgb, [1, encoded.height, encoded.width, 3])
        return (bytes.asType(.float32) / Float(255)) * Float(2) - Float(1)
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
        try WeightLoading.apply(try checkpoint.loadComponent("vae"), to: vae)
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

    /// [1, h·w, 4] rows of (t, y, x, 0), row-major like the packed latents: t is 0 for the image,
    /// 10, 20… for the pictures it is edited from.
    static func imageIds(height: Int, width: Int, t: Int32 = 0) -> MLXArray {
        var values: [Int32] = []
        values.reserveCapacity(height * width * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                values.append(contentsOf: [t, Int32(y), Int32(x), 0])
            }
        }
        return MLXArray(values, [1, height * width, 4])
    }

    /// A reference picture ([1, H, W, 3] in [-1, 1], see `KleinReference`) as the tokens that
    /// follow the image's in every pass, and their ids: the `index`-th picture sits at t = 10 + 10·index.
    public func referenceTokens(pixels: MLXArray, index: Int = 0) -> (tokens: MLXArray, ids: MLXArray) {
        let encoded = vae.encodePacked(pixels)
        eval(encoded.tokens)
        return (encoded.tokens, Self.imageIds(height: encoded.height, width: encoded.width, t: Int32(10 + 10 * index)))
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
            steps: request.steps, guidance: guidance, referencePath: request.imagePath,
            phase: phase, progress: progress, isCancelled: isCancelled
        )
    }

    /// Runs the whole pipeline for a prompt. `progress` gets each finished step; `isCancelled`
    /// is consulted between steps. A `guidance` above 1 runs mflux's classifier-free guidance
    /// against an encoded space, as its base checkpoints do. With `referencePath`, the image is
    /// edited from that picture, as `Flux2KleinEdit` does: its tokens follow the image's in every
    /// pass, and only the image's come out.
    public func generate(
        prompt: String, seed: Int, width: Int, height: Int, steps: Int, guidance: Double = 1, referencePath: String? = nil,
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
        let reference = try referencePath.map { referenceTokens(pixels: try KleinReference.pixels(path: $0)) }
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let initial = Self.initialLatents(width: width, height: height, seed: seed)
        let imageTokens = initial.latentHeight * initial.latentWidth
        let schedule = FlowMatchSchedule(steps: steps, imageSeqLen: imageTokens)
        let ids = reference.map { concatenated([initial.ids, $0.ids], axis: 1) } ?? initial.ids
        var latents = initial.latents
        for t in 0 ..< steps {
            let input = reference.map { concatenated([latents, $0.tokens], axis: 1) } ?? latents
            var noise = transformer(latents: input, prompt: encoded.embeds, timestep: schedule.timesteps[t], imageIds: ids, textIds: encoded.ids)
            if let negative {
                let negativeNoise = transformer(latents: input, prompt: negative.embeds, timestep: schedule.timesteps[t], imageIds: ids, textIds: negative.ids)
                noise = negativeNoise + Float(guidance) * (noise - negativeNoise)
            }
            if reference != nil { noise = noise[0..., 0 ..< imageTokens, 0...] }
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
