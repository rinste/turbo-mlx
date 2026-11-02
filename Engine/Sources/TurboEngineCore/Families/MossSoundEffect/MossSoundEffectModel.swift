import Foundation
import MLX
import MLXNN

public enum MossSoundEffectError: LocalizedError {
    case soundsOnly
    case noPrompt
    case duration(Double)
    case steps(Int)
    case missingFile(String, URL)

    public var errorDescription: String? {
        switch self {
        case .soundsOnly: "MOSS-SoundEffect makes sounds, not images."
        case .noPrompt: "MOSS-SoundEffect needs a description of the sound."
        case .duration(let seconds): "MOSS-SoundEffect makes sounds of 0.1 to 30 seconds; \(seconds) seconds is not."
        case .steps(let n): "MOSS-SoundEffect takes 1 to \(MossSoundEffectConfig.maxSteps) steps; \(n) is not."
        case .missingFile(let name, let url): "The MOSS-SoundEffect model at \(url.path) has no \(name)."
        }
    }

    /// The request's settings, refused by name when the model cannot run them.
    public static func validate(_ request: FamilyRequest) throws {
        guard !PythonText.strip(Array(request.prompt.unicodeScalars)).isEmpty else { throw noPrompt }
        let seconds = MossSoundEffectModel.rounded(request.duration ?? MossSoundEffectConfig.defaultSeconds)
        guard seconds > 0, seconds <= Double(MossSoundEffectConfig().maxSeconds) else { throw duration(request.duration ?? 0) }
        guard (1 ... MossSoundEffectConfig.maxSteps).contains(request.steps) else { throw steps(request.steps) }
    }
}

/// MOSS-SoundEffect v2.0 (OpenMOSS, Apache 2.0) from the MLX pack `mlx-community/MOSS-SoundEffect-v2.0-bf16` (the DiT in
/// bf16, the DAC VAE in float32, Qwen3-1.7B in bf16; `mlx/` holds the converted DiT and VAE next to the original text
/// encoder and tokenizer), run as OpenMOSS's `MossSoundEffectPipeline` runs it: the prompt with its duration through
/// Qwen3's last hidden state (512 tokens, zeros past the prompt), the flow-matching DiT over the whole 30 s of latents
/// for every clip, guidance against the negative prompt (by default empty: all zeros), the shifted Euler schedule, the
/// VAE's decode at 48 kHz mono, cut to the duration asked for. With Save memory the text encoder (4 GB) goes once
/// the prompts are encoded and is read again for the next ones.
public final class MossSoundEffectModel: AudioFamilyModel {
    public let folder: URL
    public let config: MossSoundEffectConfig
    public let transformer: MossSoundEffectTransformer
    public let vae: MossSoundEffectVAE
    public private(set) var bits: Int?
    public var lowRam = false
    private var residentTextEncoder: Qwen3TextEncoder?
    private var tokenizer: BPETokenizer?
    /// The transformer's text input [1, 512, dim] for each prompt as the encoder read it (the duration included).
    private var textCache: [String: MLXArray] = [:]
    private var cacheOrder: [String] = []

    /// Loads the pack at `folder`; `transformerFile` reads the DiT from elsewhere (OpenMOSS's float32
    /// `transformer/diffusion_pytorch_model.safetensors`, for the parity check).
    public init(folder: URL, transformerFile: URL? = nil, loadTokenizer: Bool = true) throws {
        self.folder = folder
        let ditFile = transformerFile ?? folder.appending(path: "mlx/dit.safetensors")
        let vaeFile = folder.appending(path: "mlx/vae.safetensors")
        for (name, url) in [("mlx/dit.safetensors", ditFile), ("mlx/vae.safetensors", vaeFile),
                            ("text_encoder/config.json", folder.appending(path: "text_encoder/config.json")),
                            ("tokenizer/tokenizer.json", folder.appending(path: "tokenizer/tokenizer.json"))]
        where !FileManager.default.fileExists(atPath: url.path) {
            throw MossSoundEffectError.missingFile(name, folder)
        }
        config = MossSoundEffectConfig.read(folder: folder)
        transformer = MossSoundEffectTransformer(config: config)
        let ditWeights = MossSoundEffectTransformer.keys(try loadArrays(url: ditFile))
        try WeightLoading.apply(ditWeights, to: transformer)
        bits = Self.storedBits(ditWeights, width: config.dim)
        vae = MossSoundEffectVAE(config: config)
        try WeightLoading.apply(try loadArrays(url: vaeFile), to: vae, ignoring: MossSoundEffectVAE.ignoresKey)
        eval(transformer, vae)
        residentTextEncoder = try loadTextEncoder()
        if loadTokenizer { tokenizer = try BPETokenizer(folder: folder.appending(path: "tokenizer")) }
    }

    /// The bits of a DiT stored quantized (the `-4bit` pack), from a projection's packed width.
    static func storedBits(_ weights: [String: MLXArray], width: Int) -> Int? {
        guard weights["blocks.0.self_attn.q.scales"] != nil, let packed = weights["blocks.0.self_attn.q.weight"] else { return nil }
        return packed.dim(1) * 32 / width
    }

    /// Qwen3-1.7B from `text_encoder/` (its `model.` tensors; the tied LM head is not needed).
    func loadTextEncoder() throws -> Qwen3TextEncoder {
        let encoder = Qwen3TextEncoder(config: config.textEncoder)
        var tensors: [String: MLXArray] = [:]
        for shard in try Checkpoint.shards(in: folder.appending(path: "text_encoder"), component: "") {
            for (key, value) in try loadArrays(url: shard) where key.hasPrefix("model.") {
                tensors[String(key.dropFirst("model.".count))] = value
            }
        }
        try WeightLoading.apply(tensors, to: encoder, ignoring: Qwen3TextEncoder.ignoresKey)
        eval(encoder)
        return encoder
    }

    /// The text encoder, read again if Save memory had released it.
    public func textEncoder() throws -> Qwen3TextEncoder {
        if let residentTextEncoder { return residentTextEncoder }
        let encoder = try loadTextEncoder()
        residentTextEncoder = encoder
        return encoder
    }

    /// `round(seconds, 1)`: tenths, as the pipeline formats and cuts them.
    public static func rounded(_ seconds: Double) -> Double {
        Double(String(format: "%.1f", seconds)) ?? seconds
    }

    // MARK: The prompt

    /// The token ids of a text as `WanPrompter` makes them: cleaned, tokenized without special tokens (Qwen3's add
    /// none), at most 512.
    public func tokenIds(_ text: String) throws -> [Int] {
        let tokenizer = try self.tokenizer ?? BPETokenizer(folder: folder.appending(path: "tokenizer"))
        self.tokenizer = tokenizer
        return Array(tokenizer.encode(MossSoundEffectPrompt.clean(text), addSpecialTokens: true).prefix(config.textLength))
    }

    /// `encode_prompt`: Qwen3's last hidden state (after its final norm) for the tokens, zeros for the padding up to
    /// 512 [1, 512, 2048]. The padding is masked and causal attention never reads ahead, so the tokens are run alone;
    /// an empty text is all zeros. `computeType` overrides the encoder's dtype (the parity check runs it in float32).
    public func encoderStates(ids: [Int], computeType: DType? = nil) throws -> MLXArray {
        let hidden = config.textEncoder.hiddenSize
        guard !ids.isEmpty else { return MLXArray.zeros([1, config.textLength, hidden], dtype: computeType ?? .bfloat16) }
        let encoder = try textEncoder()
        let input = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let states = encoder.hiddenStates(inputIds: input, attentionMask: MLXArray.ones([1, ids.count], dtype: .int32), computeType: computeType)
        var last = encoder.norm(states[states.count - 1])
        if ids.count < config.textLength {
            last = concatenated([last, MLXArray.zeros([1, config.textLength - ids.count, hidden], dtype: last.dtype)], axis: 1)
        }
        return last
    }

    /// The transformer's text input for a text (once; later calls read the cache).
    public func encodeText(_ text: String) throws -> MLXArray {
        if let cached = textCache[text] { return cached }
        let embedded = transformer.embedText(try encoderStates(ids: try tokenIds(text)))
        eval(embedded)
        textCache[text] = embedded
        cacheOrder.append(text)
        if cacheOrder.count > 8 { textCache[cacheOrder.removeFirst()] = nil }
        return embedded
    }

    // The prompt is read with its duration when the sound is made: the text side of `FamilyModel` does nothing.
    public func isCached(_ prompt: String) -> Bool { true }
    public func encode(_ prompt: String) throws {}
    public func promptsEncoded() {}

    // MARK: Sampling

    /// `FlowMatchScheduler.set_timesteps(steps, shift=5)` with `sigma_min=0, extra_one_step=True`, in float32 as torch
    /// works it out: `linspace(1, 0, steps + 1)[:-1]` (torch's two-sided linspace), shifted.
    public static func sigmas(steps: Int, shift: Double) -> [Float] {
        let count = steps + 1
        let step = Float(0 - 1) / Float(count - 1)
        let halfway = count / 2
        let linear = (0 ..< count).map { i -> Float in i < halfway ? 1 + step * Float(i) : 0 - step * Float(count - i - 1) }
        let s = Float(shift)
        return linear.prefix(steps).map { sigma in s * sigma / (1 + (s - 1) * sigma) }
    }

    /// The initial noise [1, 128, 1500] for a seed: drawn in float32 and rounded to bf16, as `generate_noise` casts it
    /// to the pipeline's dtype.
    public func initialNoise(seed: Int) -> MLXArray {
        MLXRandom.normal([1, config.latentChannels, config.latentFrames], key: MLXRandom.key(UInt64(truncatingIfNeeded: seed)))
            .asType(.bfloat16).asType(.float32)
    }

    /// The guided velocity in float32: `u + g·(c − u)` when guidance is not 1, the prompt's alone when it is.
    public func velocity(latents: MLXArray, timestep: Float, prompt: MLXArray, negative: MLXArray, guidance: Float) -> MLXArray {
        let (time, modulation) = transformer.timeEmbedding(Double(timestep))
        let conditional = transformer(latents, time: time, modulation: modulation, text: prompt).asType(.float32)
        guard guidance != 1 else { return conditional }
        let unconditional = transformer(latents, time: time, modulation: modulation, text: negative).asType(.float32)
        return unconditional + guidance * (conditional - unconditional)
    }

    /// The denoising loop from `noise` [1, 128, T] (float32): one Euler step per scheduled sigma, the last to 0.
    public func denoise(
        noise: MLXArray, prompt: MLXArray, negative: MLXArray, steps: Int, guidance: Float,
        progress: (Int, Int) -> Void = { _, _ in }, isCancelled: () -> Bool = { false }
    ) throws -> MLXArray {
        let sigmas = Self.sigmas(steps: steps, shift: config.shift)
        var latents = noise.asType(.float32)
        for index in 0 ..< steps {
            let timestep = sigmas[index] * Float(config.trainTimesteps)
            let velocity = velocity(latents: latents, timestep: timestep, prompt: prompt, negative: negative, guidance: guidance)
            let next = index + 1 < steps ? sigmas[index + 1] : 0
            latents = latents + velocity * (next - sigmas[index])
            eval(latents)
            progress(index + 1, steps)
            if isCancelled() { throw GenerationError.cancelled }
        }
        return latents
    }

    /// Latent frames past the kept ones that the decode still reads: more than the decoder's reach (its 7-wide
    /// convolutions, dilated up to 9 at each rate, see a few latents to either side), so the kept samples are those of
    /// the whole decode.
    static let decodeMargin = 16

    /// The waveform [samples] of latents [1, 128, T], cut to `seconds` (the reference's `int(48000 · seconds)`). The
    /// reference decodes all 30 s and cuts; only the latents the kept samples depend on are decoded here.
    public func decode(_ latents: MLXArray, seconds: Double) -> MLXArray {
        let samples = min(Int(Double(config.sampleRate) * Self.rounded(seconds)), latents.dim(2) * config.hopLength)
        let frames = min(latents.dim(2), (samples + config.hopLength - 1) / config.hopLength + Self.decodeMargin)
        return vae.decode(latents[0..., 0..., 0 ..< frames])[0, 0 ..< samples]
    }

    public func generate(_ request: FamilyRequest, phase: (GenerationPhase) -> Void, progress: (Int, Int) -> Void,
                         isCancelled: () -> Bool) throws -> GeneratedImage {
        throw MossSoundEffectError.soundsOnly
    }

    public func generateAudio(
        _ request: FamilyRequest, phase: (AudioPhase) -> Void, progress: (Int, Int) -> Void, isCancelled: () -> Bool
    ) throws -> GeneratedAudio {
        try MossSoundEffectError.validate(request)
        let seconds = Self.rounded(request.duration ?? MossSoundEffectConfig.defaultSeconds)
        let guidance = Float(request.guidance)

        phase(.encoding)
        let prompt = try encodeText(MossSoundEffectPrompt.withDuration(request.prompt, seconds: seconds))
        // Without guidance there is no unconditional pass to read a negative prompt.
        let negative = guidance != 1 ? try encodeText(request.negativePrompt ?? "") : prompt
        if lowRam, residentTextEncoder != nil {
            residentTextEncoder = nil
            Memory.clearCache()
        }
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising(total: request.steps))
        let latents = try denoise(noise: initialNoise(seed: request.seed), prompt: prompt, negative: negative, steps: request.steps,
                                  guidance: guidance, progress: progress, isCancelled: isCancelled)

        phase(.decoding)
        let wave = clip(decode(latents, seconds: seconds), min: -1, max: 1).expandedDimensions(axis: 1)
        eval(wave)
        return GeneratedAudio(samples: wave, sampleRate: config.sampleRate, frames: Int((seconds * 50).rounded(.up)))
    }
}
