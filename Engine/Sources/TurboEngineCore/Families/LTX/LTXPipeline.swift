import Foundation
import MLX
import MLXNN

/// What a video family produces besides the frames and sound it hands to the writer.
public struct GeneratedClip {
    public let width: Int
    public let height: Int
    public let frames: Int
    public let fps: Double
}

/// A family that makes clips. The sound goes to `audio` first ([2, samples] float32 at the given
/// rate, or nil), then the frames to `frames` as they are decoded ([N, H, W, 3] uint8, in order):
/// the writer interleaves the two.
public protocol VideoFamilyModel: FamilyModel {
    /// The steps `progress` will count to for this request.
    func totalSteps(_ request: FamilyRequest) -> Int
    func generateVideo(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool,
        audio: (MLXArray?, Int) throws -> Void,
        frames: (MLXArray) throws -> Void
    ) throws -> GeneratedClip
}

public enum LTXError: LocalizedError {
    case videoOnly
    case missingFile(String, URL)
    case unreadableImage(String)

    public var errorDescription: String? {
        switch self {
        case .videoOnly: "LTX-2 makes videos, not still images."
        case .missingFile(let name, let url): "The LTX-2 model at \(url.path) has no \(name)."
        case .unreadableImage(let path): "Could not read the reference image \(path)."
        }
    }
}

/// A latent being denoised with the conditioning that pins some of its tokens
/// (`LatentState`: the tokens, their clean values, and 1 where they are generated).
struct LTXLatentState {
    var latent: MLXArray
    var clean: MLXArray
    var mask: MLXArray
    /// All tokens generated: the transformer then runs one timestep for the whole stream.
    var uniform: Bool
}

/// LTX-2.3 distilled, the two-stage pipeline of dgrauet's `DistilledPipeline`: the prompt through
/// Gemma 3 and the connector, eight steps at half resolution, the latent upsampled ×2, three
/// steps at full resolution, then the video and audio decoders. A reference image pins the first
/// frame (`VideoConditionByLatentIndex`) in both stages.
public final class LTXVideoModel: VideoFamilyModel {
    public let config: LTXConfig
    public let pack: URL
    public let textEncoderFolder: URL
    public private(set) var bits: Int?
    public var lowRam = false

    private var prompter: Gemma3Prompter?
    private var textSide: (gemma: Gemma3TextModel, connector: LTXTextConnector)?
    private var transformer: LTXTransformer?
    /// The transformer has generated since it was loaded: its weights are resident.
    private var transformerUsed = false
    private var encoderStatistics: LTXEncoderStatistics?
    /// Embeddings per prompt: video [1, T, 4096] and audio [1, T, 2048].
    private var promptCache: [String: (video: MLXArray, audio: MLXArray)] = [:]

    /// `pack` is a dgrauet LTX-2.3 folder; `textEncoder` the Gemma 3 12B folder.
    public init(pack: URL, textEncoder: URL, loadTokenizer: Bool = true) throws {
        self.pack = pack
        textEncoderFolder = textEncoder
        config = LTXConfig.load(pack: pack)
        if loadTokenizer {
            prompter = try Gemma3Prompter(folder: textEncoder, maxLength: LTXConfig.maxPromptTokens)
        }
        try loadTransformer()
    }

    // MARK: Loading

    private func file(_ name: String) throws -> URL {
        let url = pack.appending(path: name)
        guard FileManager.default.fileExists(atPath: url.path) else { throw LTXError.missingFile(name, pack) }
        return url
    }

    /// The distilled transformer, the latest version the pack has (`transformer-distilled-1.1`).
    private func transformerFile() throws -> URL {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: pack.path)) ?? []
        let versioned = names.filter { $0.hasPrefix("transformer-distilled-") && $0.hasSuffix(".safetensors") }.sorted()
        if let latest = versioned.last { return pack.appending(path: latest) }
        return try file("transformer-distilled.safetensors")
    }

    private func loadTransformer() throws {
        let model = LTXTransformer(config: config.transformer)
        let tensors = try loadArrays(url: transformerFile())
        let weights = LTXTransformer.weights(tensors)
        try WeightLoading.apply(weights, to: model)
        bits = Self.storedBits(weights)
        transformer = model
        transformerUsed = false
    }

    /// Bits of the first quantized block linear (the pack quantizes only those).
    private static func storedBits(_ weights: [String: MLXArray]) -> Int? {
        guard let scales = weights["transformer_blocks.0.attn1.to_q.scales"],
              let packed = weights["transformer_blocks.0.attn1.to_q.weight"] else { return nil }
        let input = scales.shape[1] * 64
        return packed.shape[1] * 32 / input
    }

    private func loadedTransformer() throws -> LTXTransformer {
        if let transformer { return transformer }
        try loadTransformer()
        return transformer!
    }

    private func loadTextSide() throws {
        let gemmaConfig = try Gemma3Config.load(folder: textEncoderFolder)
        let gemma = Gemma3TextModel(config: gemmaConfig)
        var tensors: [String: MLXArray] = [:]
        let names = (try? FileManager.default.contentsOfDirectory(atPath: textEncoderFolder.path)) ?? []
        for name in names.sorted() where name.hasSuffix(".safetensors") {
            tensors.merge(try loadArrays(url: textEncoderFolder.appending(path: name))) { _, new in new }
        }
        try WeightLoading.apply(Gemma3TextModel.weights(tensors), to: gemma)
        let connector = LTXTextConnector(config: config)
        try WeightLoading.apply(LTXTextConnector.weights(try loadArrays(url: file("connector.safetensors"))), to: connector)
        textSide = (gemma, connector)
    }

    /// Gemma and the connector, reloaded if they had been released. In low-RAM mode a resident
    /// transformer goes first, so the two sides are never in memory together.
    private func loadedTextSide() throws -> (gemma: Gemma3TextModel, connector: LTXTextConnector) {
        if let textSide { return textSide }
        if lowRam, transformerUsed {
            transformer = nil
            Memory.clearCache()
        }
        try loadTextSide()
        return textSide!
    }

    /// The video encoder's latent statistics, which the upsampling step (de)normalizes with.
    private func statistics() throws -> LTXEncoderStatistics {
        if let encoderStatistics { return encoderStatistics }
        let module = LTXEncoderStatistics(channels: LTXConfig.latentChannels)
        let tensors = LTXVideoEncoder.weights(try loadArrays(url: file("vae_encoder.safetensors")))
        try WeightLoading.apply(stripping("per_channel_statistics.", from: tensors), to: module)
        encoderStatistics = module
        return module
    }

    private func loadModule<M: Module>(_ module: M, file name: String, weights: ([String: MLXArray]) -> [String: MLXArray]) throws -> M {
        try WeightLoading.apply(weights(try loadArrays(url: file(name))), to: module)
        return module
    }

    // MARK: Prompts

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil }

    public func encode(_ prompt: String) throws {
        guard promptCache[prompt] == nil else { return }
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(textEncoderFolder) }
        let (gemma, connector) = try loadedTextSide()
        let (tokens, mask) = prompter.tokenize(prompt)
        let states = gemma.allHiddenStates(tokens: tokens, attentionMask: mask)
        let (video, audio) = connector(hiddenStates: states, attentionMask: mask)
        eval(video, audio)
        promptCache[prompt] = (video, audio)
    }

    public func promptsEncoded() {
        if lowRam, textSide != nil {
            textSide = nil
            Memory.clearCache()
        }
    }

    /// The text side's output for a prompt (for `verify`).
    public func embeddings(_ prompt: String) throws -> (video: MLXArray, audio: MLXArray) {
        try encode(prompt)
        return promptCache[prompt]!
    }

    // MARK: Generation

    public func generate(
        _ request: FamilyRequest, phase: (GenerationPhase) -> Void, progress: (Int, Int) -> Void, isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        throw LTXError.videoOnly
    }

    public func totalSteps(_ request: FamilyRequest) -> Int {
        Self.shortened(LTXConfig.distilledSigmas, steps: request.steps).count - 1 + LTXConfig.stage2Sigmas.count - 1
    }

    public func generateVideo(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool,
        audio audioSink: (MLXArray?, Int) throws -> Void,
        frames sink: (MLXArray) throws -> Void
    ) throws -> GeneratedClip {
        // Every step allocates the same buffers: a few GB of cache reuse them, and more would keep
        // the memory of earlier passes (15 GB more than the 38 GB in use with the 8-bit pack).
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = min(previousLimit, 4 << 30)
        defer { Memory.cacheLimit = previousLimit }
        let geometry = LTXGeometry(width: request.width, height: request.height, frames: request.frames ?? 121, fps: request.fps ?? 24)
        let stage1Sigmas = Self.shortened(LTXConfig.distilledSigmas, steps: request.steps)
        let stage2Sigmas = LTXConfig.stage2Sigmas
        let totalSteps = stage1Sigmas.count - 1 + stage2Sigmas.count - 1

        phase(.encoding)
        try encode(request.prompt)
        promptsEncoded()
        guard let (videoText, audioText) = promptCache[request.prompt] else { throw GenerationError.cancelled }
        if isCancelled() { throw GenerationError.cancelled }

        let transformer = try loadedTransformer()
        let reference = try request.imagePath.map { try LTXReferenceImage(path: $0) }
        var encoder: LTXVideoEncoder? = reference == nil ? nil
            : try loadModule(LTXVideoEncoder(), file: "vae_encoder.safetensors", weights: LTXVideoEncoder.weights)

        // Stage 1: half resolution, from noise.
        phase(.denoising)
        let f = geometry.latentFrames
        let (h1, w1) = (geometry.latentHeight / 2, geometry.latentWidth / 2)
        let audioTokens = geometry.audioTokens
        var video1 = Self.initialState(shape: [1, f * h1 * w1, 128], seed: request.seed)
        if let reference, let encoder {
            let tokens = reference.tokens(width: w1 * 32, height: h1 * 32, encoder: encoder)
            video1 = Self.conditioned(video1, firstFrame: tokens, tokensPerFrame: h1 * w1)
        }
        let audio1 = Self.initialState(shape: [1, audioTokens, 128], seed: request.seed + 1)
        var step = 0
        let (videoHalf, audioLatent) = try denoise(
            transformer: transformer, video: video1, audio: audio1, sigmas: stage1Sigmas,
            videoText: videoText, audioText: audioText,
            videoPositions: Self.videoPositions(frames: f, height: h1, width: w1, fps: geometry.fps),
            audioPositions: Self.audioPositions(audioTokens),
            onStep: { step += 1; progress(step, totalSteps) }, isCancelled: isCancelled
        )
        transformerUsed = true

        // The latent upsampled ×2, in the encoder's un-normalized space.
        let upsampled = try upsample(videoHalf, frames: f, height: h1, width: w1)
        if isCancelled() { throw GenerationError.cancelled }

        // Stage 2: full resolution, the upsampled latent renoised at the table's first sigma.
        let (h2, w2) = (h1 * 2, w1 * 2)
        let start = stage2Sigmas[0]
        var video2 = Self.renoised(upsampled, sigma: start, seed: request.seed + LTXConfig.stage2SeedOffset)
        if let reference, let encoder {
            let tokens = reference.tokens(width: w2 * 32, height: h2 * 32, encoder: encoder)
            video2 = Self.conditioned(video2, firstFrame: tokens, tokensPerFrame: h2 * w2)
        }
        encoder = nil
        let audio2 = Self.renoisedMasked(audioLatent, sigma: start, seed: request.seed + LTXConfig.stage2SeedOffset)
        let (videoFull, audioFull) = try denoise(
            transformer: transformer, video: video2, audio: audio2, sigmas: stage2Sigmas,
            videoText: videoText, audioText: audioText,
            videoPositions: Self.videoPositions(frames: f, height: h2, width: w2, fps: geometry.fps),
            audioPositions: Self.audioPositions(audioTokens),
            onStep: { step += 1; progress(step, totalSteps) }, isCancelled: isCancelled
        )

        // The decoders, the transformer out of the way in low-RAM mode.
        phase(.decoding)
        if lowRam {
            self.transformer = nil
            Memory.clearCache()
        }
        // The sound first (small), so the writer can interleave it with the frames.
        try audioSink(try decodeAudio(audioFull), 48000)
        let latent = videoFull.reshaped([1, f, h2, w2, 128]).transposed(0, 4, 1, 2, 3)
        try decodeVideo(latent, fps: geometry.fps, isCancelled: isCancelled, sink: sink)

        return GeneratedClip(width: w2 * 32, height: h2 * 32, frames: geometry.frames, fps: geometry.fps)
    }

    /// `shorten_schedule(keep="start")`: σ = 1, then the last `steps` sigmas of the table.
    static func shortened(_ table: [Double], steps: Int) -> [Double] {
        guard steps > 0, steps < table.count - 1 else { return table }
        return [table[0]] + table.suffix(steps)
    }

    // MARK: States

    /// `mx.random.seed(seed); mx.random.normal(shape)`: the global key split once.
    static func seededNormal(_ shape: [Int], seed: Int) -> MLXArray {
        let key = MLXRandom.split(key: MLXRandom.key(UInt64(seed))).1
        return MLXRandom.normal(shape, key: key)
    }

    /// Stage 1: pure noise in bfloat16 (`legacy_scalar_blend` at σ = 1 over a zero latent).
    static func initialState(shape: [Int], seed: Int) -> LTXLatentState {
        let noise = seededNormal(shape, seed: seed).asType(.bfloat16)
        let latent = noise * Float(1) + MLXArray.zeros(shape, dtype: .bfloat16) * Float(0)
        return LTXLatentState(latent: latent, clean: MLXArray.zeros(shape, dtype: .bfloat16),
                              mask: MLXArray.ones([shape[0], shape[1], 1], dtype: .bfloat16), uniform: true)
    }

    /// Stage 2 video: `noise·σ + latent·(1 − σ)` with the scalar σ (`legacy_scalar_blend`).
    static func renoised(_ latent: MLXArray, sigma: Double, seed: Int) -> LTXLatentState {
        let noise = seededNormal(latent.shape, seed: seed).asType(.bfloat16)
        let blended = noise * Float(sigma) + latent * Float(1 - sigma)
        return LTXLatentState(latent: blended, clean: latent,
                              mask: MLXArray.ones([latent.shape[0], latent.shape[1], 1], dtype: .bfloat16), uniform: true)
    }

    /// Stage 2 audio: `noise_latent_state`, σ taken through the bfloat16 mask.
    static func renoisedMasked(_ latent: MLXArray, sigma: Double, seed: Int) -> LTXLatentState {
        let noise = seededNormal(latent.shape, seed: seed).asType(latent.dtype)
        let mask = MLXArray.ones([latent.shape[0], latent.shape[1], 1], dtype: .bfloat16)
        let scaled = mask * Float(sigma)
        let blended = noise * scaled + latent * (1 - scaled)
        return LTXLatentState(latent: blended, clean: latent, mask: mask, uniform: true)
    }

    /// `VideoConditionByLatentIndex` at frame 0: the first frame's tokens replaced by the image's,
    /// in the latent and the clean latent, and preserved (mask 0, a float32 mask from then on).
    static func conditioned(_ state: LTXLatentState, firstFrame tokens: MLXArray, tokensPerFrame: Int) -> LTXLatentState {
        let count = state.latent.shape[1]
        let rest = tokensPerFrame ..< count
        let latent = concatenated([tokens.asType(state.latent.dtype), state.latent[0..., rest, 0...]], axis: 1)
        let clean = concatenated([tokens.asType(state.clean.dtype), state.clean[0..., rest, 0...]], axis: 1)
        let mask = concatenated([MLXArray.zeros([1, tokensPerFrame, 1], dtype: .float32), state.mask[0..., rest, 0...]], axis: 1)
        return LTXLatentState(latent: latent, clean: clean, mask: mask, uniform: false)
    }

    /// `compute_video_positions`: [1, F·H·W, 3] float32, the time of each latent frame's middle
    /// in seconds (the first frame covers one pixel frame) and the pixel centers of its cells.
    static func videoPositions(frames: Int, height: Int, width: Int, fps: Double) -> MLXArray {
        let t = LTXConfig.temporalScale
        let s = LTXConfig.spatialScale
        let index = MLXArray(0 ..< frames).asType(.float32)
        let starts = maximum(index * Float(t) + Float(1 - t), MLXArray(Float(0)))
        let ends = maximum((index + 1) * Float(t) + Float(1 - t), MLXArray(Float(0)))
        let middles = (starts + ends) / 2 / Float(fps)
        let rows = MLXArray(0 ..< height).asType(.float32) * Float(s) + Float(s) / 2
        let columns = MLXArray(0 ..< width).asType(.float32) * Float(s) + Float(s) / 2
        let timeGrid = broadcast(middles.reshaped([frames, 1, 1]), to: [frames, height, width])
        let rowGrid = broadcast(rows.reshaped([1, height, 1]), to: [frames, height, width])
        let columnGrid = broadcast(columns.reshaped([1, 1, width]), to: [frames, height, width])
        return stacked([timeGrid, rowGrid, columnGrid], axis: -1).reshaped([1, frames * height * width, 3])
    }

    /// `compute_audio_positions`: [1, T, 1], each latent's middle in seconds.
    static func audioPositions(_ count: Int) -> MLXArray {
        let index = MLXArray(0 ..< count).asType(.float32)
        let starts = maximum(index * Float(4) + Float(-3), MLXArray(Float(0))) * Float(160) / Float(16000)
        let ends = maximum((index + 1) * Float(4) + Float(-3), MLXArray(Float(0))) * Float(160) / Float(16000)
        return ((starts + ends) / 2).reshaped([1, count, 1])
    }

    // MARK: Denoising

    /// `denoise_loop`: x0 from the velocity, the preserved tokens put back, an Euler step.
    private func denoise(
        transformer: LTXTransformer, video: LTXLatentState, audio: LTXLatentState, sigmas: [Double],
        videoText: MLXArray, audioText: MLXArray, videoPositions: MLXArray, audioPositions: MLXArray,
        onStep: () -> Void, isCancelled: () -> Bool
    ) throws -> (video: MLXArray, audio: MLXArray) {
        var videoX = video.latent
        var audioX = audio.latent
        for index in 0 ..< sigmas.count - 1 {
            let (sigma, next) = (sigmas[index], sigmas[index + 1])
            let sigmaArray = MLXArray([Float(sigma)]).asType(.bfloat16)
            let videoTimesteps: MLXArray? = video.uniform ? nil : (video.mask * Float(sigma)).squeezed(axis: -1)
            let (videoVelocity, audioVelocity) = transformer(
                video: videoX, audio: audioX, sigma: sigmaArray, videoTimesteps: videoTimesteps,
                videoText: videoText, audioText: audioText, videoPositions: videoPositions, audioPositions: audioPositions
            )
            let videoSigma = (videoTimesteps?.expandedDimensions(axis: -1) ?? sigmaArray.reshaped([1, 1, 1])).asType(.float32)
            let audioSigma = sigmaArray.reshaped([1, 1, 1]).asType(.float32)
            var videoX0 = (videoX.asType(.float32) - videoSigma * videoVelocity.asType(.float32)).asType(videoX.dtype)
            var audioX0 = (audioX.asType(.float32) - audioSigma * audioVelocity.asType(.float32)).asType(audioX.dtype)
            videoX0 = videoX0 * video.mask + video.clean * (1 - video.mask)
            audioX0 = audioX0 * audio.mask + audio.clean * (1 - audio.mask)
            videoX = Self.eulerStep(videoX, x0: videoX0, sigma: sigma, next: next)
            audioX = Self.eulerStep(audioX, x0: audioX0, sigma: sigma, next: next)
            eval(videoX, audioX)
            onStep()
            if isCancelled() { throw GenerationError.cancelled }
        }
        return (videoX, audioX)
    }

    static func eulerStep(_ x: MLXArray, x0: MLXArray, sigma: Double, next: Double) -> MLXArray {
        guard sigma != 0 else { return x0 }
        let derivative = (x - x0) / Float(sigma)
        return x + Float(next - sigma) * derivative
    }

    // MARK: Upsampling and decoding

    /// `_upsample_latent`: stage-1 tokens [1, F·h·w, 128] → normalized upsampled latent tokens.
    private func upsample(_ tokens: MLXArray, frames: Int, height: Int, width: Int) throws -> MLXArray {
        let stats = try statistics()
        let upsampler = try loadModule(LTXLatentUpsampler(), file: "spatial_upscaler_x2_v1_1.safetensors",
                                       weights: LTXLatentUpsampler.weights)
        let latent = tokens.reshaped([1, frames, height, width, 128])
        let denormalized = stats.denormalize(latent).transposed(0, 4, 1, 2, 3)
        let upscaled = upsampler(denormalized).transposed(0, 2, 3, 4, 1)
        let normalized = stats.normalize(upscaled)
        eval(normalized)
        return normalized.reshaped([1, -1, 128])
    }

    /// Latent [1, 128, F, H, W] → frames handed to `sink` in order, decoded in tiles when the whole
    /// clip would not fit the decode budget.
    private func decodeVideo(_ latent: MLXArray, fps: Double, isCancelled: () -> Bool, sink: (MLXArray) throws -> Void) throws {
        let decoder = try loadModule(LTXVideoDecoder(), file: "vae_decoder.safetensors", weights: LTXVideoDecoder.weights)
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = previousLimit }
        let tiling = LTXDecodeTiling.plan(latentShape: latent.shape, fps: fps, budget: LTXDecodeTiling.budget(resident: Memory.activeMemory))
        try LTXDecodeTiling.decode(latent, decoder: decoder, tiling: tiling, isCancelled: isCancelled) { chunk in
            try sink(Self.toFrames(chunk))
        }
    }

    /// [1, 3, N, H, W] in [-1, 1] → [N, H, W, 3] uint8, truncated as the reference converts it.
    static func toFrames(_ chunk: MLXArray) -> MLXArray {
        let clipped = clip(chunk, min: -1, max: 1)
        let bytes = ((clipped + 1) * Float(127.5)).asType(.uint8)
        return bytes[0].transposed(1, 2, 3, 0)
    }

    // MARK: Stages, for verify (LTXVerification.swift)

    var prompterForVerification: Gemma3Prompter? { prompter }
    func textModels() throws -> (gemma: Gemma3TextModel, connector: LTXTextConnector) { try loadedTextSide() }
    func transformerForVerification() throws -> LTXTransformer { try loadedTransformer() }
    func denoiseForVerification(
        video: LTXLatentState, audio: LTXLatentState, sigmas: [Double], videoText: MLXArray, audioText: MLXArray,
        videoPositions: MLXArray, audioPositions: MLXArray
    ) throws -> (video: MLXArray, audio: MLXArray) {
        try denoise(transformer: try loadedTransformer(), video: video, audio: audio, sigmas: sigmas,
                    videoText: videoText, audioText: audioText, videoPositions: videoPositions, audioPositions: audioPositions,
                    onStep: {}, isCancelled: { false })
    }
    func upsampleForVerification(_ tokens: MLXArray, frames: Int, height: Int, width: Int) throws -> MLXArray {
        try upsample(tokens, frames: frames, height: height, width: width)
    }
    func encoderForVerification() throws -> LTXVideoEncoder {
        try loadModule(LTXVideoEncoder(), file: "vae_encoder.safetensors", weights: LTXVideoEncoder.weights)
    }
    func videoDecoderForVerification() throws -> LTXVideoDecoder {
        try loadModule(LTXVideoDecoder(), file: "vae_decoder.safetensors", weights: LTXVideoDecoder.weights)
    }
    func audioDecoderForVerification() throws -> LTXAudioDecoder {
        try loadModule(LTXAudioDecoder(), file: "audio_vae.safetensors", weights: LTXAudioDecoder.weights)
    }
    func vocoderForVerification() throws -> LTXVocoder {
        try loadModule(LTXVocoder(), file: "vocoder.safetensors", weights: LTXVocoder.weights)
    }

    /// Audio latent tokens [1, T, 128] → 48 kHz stereo [2, samples] float32.
    private func decodeAudio(_ tokens: MLXArray) throws -> MLXArray {
        let decoder = try loadModule(LTXAudioDecoder(), file: "audio_vae.safetensors", weights: LTXAudioDecoder.weights)
        let vocoder = try loadModule(LTXVocoder(), file: "vocoder.safetensors", weights: LTXVocoder.weights)
        let count = tokens.shape[1]
        let latent = tokens.reshaped([1, count, 8, 16]).transposed(0, 2, 1, 3)
        let mel = decoder.decode(latent)
        let waveform = vocoder(mel)
        eval(waveform)
        return waveform[0].asType(.float32)
    }
}
