import Foundation
import MLX
import MLXNN

/// What a family making sound reports while it works, in order.
public enum AudioPhase {
    case encoding
    /// The language models write the song frame by frame, up to `frames`; they may end it sooner.
    case composing(frames: Int)
    case denoising(total: Int)
    case decoding
}

/// One song: [samples, 2] float32 in −1…1.
public struct GeneratedAudio {
    public let samples: MLXArray
    public let sampleRate: Int
    /// The language models' frames, `frameRate` a second.
    public let frames: Int
    public var seconds: Double { Double(samples.dim(0)) / Double(sampleRate) }
}

/// A family that makes sound from a description and lyrics (MiniMax Music 3).
public protocol AudioFamilyModel: FamilyModel {
    func generateAudio(
        _ request: FamilyRequest,
        phase: (AudioPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedAudio
}

public enum Music3Error: LocalizedError {
    case songsOnly
    case noCaption
    case noLyrics
    case duration(Double)
    case steps(Int)
    case promptTooLong(Int)
    case noFrames
    case missingFile(String, URL)

    public var errorDescription: String? {
        switch self {
        case .songsOnly: "MiniMax-Music3 makes songs, not images."
        case .noCaption: "MiniMax-Music3 needs a description of the music."
        case .noLyrics: "MiniMax-Music3 needs lyrics; write [instrumental] for a song without words."
        case .duration(let seconds): "MiniMax-Music3 makes songs of up to 6 minutes; \(seconds) seconds is not between 0 and 360."
        case .steps(let n): "MiniMax-Music3 takes 1 to 30 steps; \(n) is not."
        case .promptTooLong(let n): "The description and lyrics make \(n) tokens; MiniMax-Music3 reads at most 5,000."
        case .noFrames: "MiniMax-Music3 ended the song before it began: try another seed or description."
        case .missingFile(let name, let url): "The MiniMax-Music3 model at \(url.path) has no \(name)."
        }
    }

    /// The request's settings, refused by name when the model cannot run them.
    public static func validate(_ request: FamilyRequest) throws {
        guard !PythonText.strip(Array(request.prompt.unicodeScalars)).isEmpty else { throw noCaption }
        guard let lyrics = request.lyrics, !PythonText.strip(Array(lyrics.unicodeScalars)).isEmpty else { throw noLyrics }
        let duration = request.duration ?? Music3Model.defaultDuration
        guard duration > 0, duration <= Music3Config.maxDuration else { throw Music3Error.duration(duration) }
        guard (1 ... Music3Config.maxSteps).contains(request.steps) else { throw Music3Error.steps(request.steps) }
    }
}

/// MiniMax Music 3 (MiniMax-Music3 Community License) from mlx-audio's MLX pack (`mlx-community/MiniMax-Music3-8bit`:
/// the language models and the flow transformer in 8 bits, the rest bf16), run as mlx-audio runs it: the prompt's
/// tokens, the hierarchical autoregressive frames, the guided flow on windows of 200 frames, the vocoder at 44.1 kHz
/// stereo. With Save memory, the language models (11 GB) go once the frames are written and are read again for the
/// next song.
public final class Music3Model: AudioFamilyModel {
    public static let defaultDuration = 60.0

    public let folder: URL
    public let config: Music3Config
    public let conditionEncoder: Music3ConditionEncoder
    public let transformer: Music3FlowTransformer
    public let vocoder: Music3Vocoder
    private var resident: (languageModel: Music3LanguageModel, depth: Music3DepthDecoder)?
    private var tokenizer: BPETokenizer?
    public var lowRam = false
    public var bits: Int? { 8 }

    public init(folder: URL, loadTokenizer: Bool = true) throws {
        self.folder = folder
        for name in ["config.json", "model.safetensors.index.json", "tokenizer/tokenizer.json"]
        where !FileManager.default.fileExists(atPath: folder.appending(path: name).path) {
            throw Music3Error.missingFile(name, folder)
        }
        config = try Music3Config.read(folder: folder)
        let weights = try Self.weights(folder: folder, components: ["condition_encoder", "transformer", "vocoder"])
        conditionEncoder = try Self.load(Music3ConditionEncoder(config: config), weights["condition_encoder"])
        transformer = try Self.load(Music3FlowTransformer(config: config), weights["transformer"])
        vocoder = try Self.load(Music3Vocoder(config: config), weights["vocoder"])
        resident = try loadLanguageModels()
        if loadTokenizer { tokenizer = try BPETokenizer(folder: folder.appending(path: "tokenizer")) }
    }

    /// The pack's tensors of `components`, keyed by component, their prefix removed.
    static func weights(folder: URL, components: Set<String>) throws -> [String: [String: MLXArray]] {
        var grouped: [String: [String: MLXArray]] = [:]
        for shard in try Checkpoint.shards(in: folder, component: "") {
            for (key, value) in try loadArrays(url: shard) {
                guard let dot = key.firstIndex(of: "."), components.contains(String(key[..<dot])) else { continue }
                grouped[String(key[..<dot]), default: [:]][String(key[key.index(after: dot)...])] = value
            }
        }
        return grouped
    }

    static func load<M: Module>(_ module: M, _ tensors: [String: MLXArray]?) throws -> M {
        try WeightLoading.apply(tensors ?? [:], to: module)
        eval(module)
        return module
    }

    private func loadLanguageModels() throws -> (Music3LanguageModel, Music3DepthDecoder) {
        let weights = try Self.weights(folder: folder, components: ["language_model", "rvq_depth_decoder"])
        return (try Self.load(Music3LanguageModel(config: config), weights["language_model"]),
                try Self.load(Music3DepthDecoder(config: config), weights["rvq_depth_decoder"]))
    }

    /// The prompt's ids, conditional and unconditional, [2, tokens].
    public func promptIDs(caption: String, lyrics: String) throws -> MLXArray {
        let tokenizer = try self.tokenizer ?? BPETokenizer(folder: folder.appending(path: "tokenizer"))
        self.tokenizer = tokenizer
        let ids = tokenizer.encode(Music3Prompt.assemble(caption: caption, lyrics: lyrics), addSpecialTokens: true)
        guard ids.count <= Music3Config.maxPromptTokens else { throw Music3Error.promptTooLong(ids.count) }
        let pair = Music3Prompt.idPair(ids, cfgToken: config.audioCFGTokenID)
        return MLXArray((pair.conditional + pair.unconditional).map(Int32.init)).reshaped(2, ids.count)
    }

    /// The language models, read again if Save memory had released them.
    public func languageModels() throws -> Music3AR {
        let models = try resident ?? loadLanguageModels()
        resident = models
        return Music3AR(languageModel: models.languageModel, depth: models.depth, config: config)
    }

    /// The frames' hidden states for a prompt.
    public func frames(ids: MLXArray, maxFrames: Int, seed: Int, progress: (Int) -> Void = { _ in },
                       isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let frames = try languageModels().generate(ids: ids, maxFrames: maxFrames, seed: seed, progress: progress, isCancelled: isCancelled)
        eval(frames)
        if lowRam {
            resident = nil
            Memory.clearCache()
        }
        return frames
    }

    /// Every window's latents (`_run_flow` without the vocoder), the noise drawn from seed + 7.
    public func latents(frames: MLXArray, steps: Int, seed: Int, progress: (Int, Int) -> Void = { _, _ in },
                        isCancelled: () -> Bool = { false }) throws -> [MLXArray] {
        let flow = Music3Flow(conditionEncoder: conditionEncoder, transformer: transformer, config: config)
        let starts = Music3Flow.chunkStarts(frames: frames.dim(1))
        let noise = MLXRandom.RandomState(seed: UInt64(truncatingIfNeeded: seed + 7))
        var previous: (latent: MLXArray, condition: MLXArray)?
        var chunks: [MLXArray] = []
        var done = 0
        for start in starts {
            let end = min(start + Music3Config.chunkFrames, frames.dim(1))
            let condition = conditionEncoder(frames[0..., start ..< end])
            let initial = MLXRandom.normal([1, config.ditInChannels, condition.dim(1)], key: noise).asType(condition.dtype)
            let (latents, merged) = try flow.denoise(noise: initial, condition: condition, steps: steps, previous: previous) {
                done += 1
                progress(done, starts.count * steps)
                if isCancelled() { throw GenerationError.cancelled }
            }
            let carryStart = max(0, latents.dim(-1) - 2 * Music3Config.overlapLatents)
            let carryEnd = max(carryStart, latents.dim(-1) - Music3Config.overlapLatents)
            previous = (latents[.ellipsis, carryStart ..< carryEnd], merged[0..., carryStart ..< carryEnd])
            chunks.append(latents)
        }
        return chunks
    }

    /// The song's waveform [1, 2, samples]: each window decoded, the middle of each kept.
    public func decode(_ chunks: [MLXArray], isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let hop = config.outputHopLength
        var pieces: [MLXArray] = []
        for (index, latents) in chunks.enumerated() {
            let wave = vocoder(latents)
            let left = index == 0 ? 0 : Music3Config.cropLeftLatents * hop
            let right = index == chunks.count - 1 ? 0 : Music3Config.cropRightLatents * hop
            let end = wave.dim(-1) - right
            let kept = end <= left ? wave : wave[.ellipsis, left ..< end]
            eval(kept)
            pieces.append(kept)
            if isCancelled() { throw GenerationError.cancelled }
        }
        return concatenated(pieces, axis: -1)
    }

    // The prompt is read with the lyrics when the song is made: the text side of `FamilyModel` does nothing.
    public func isCached(_ prompt: String) -> Bool { true }
    public func encode(_ prompt: String) throws {}
    public func promptsEncoded() {}

    public func generate(_ request: FamilyRequest, phase: (GenerationPhase) -> Void, progress: (Int, Int) -> Void,
                         isCancelled: () -> Bool) throws -> GeneratedImage {
        throw Music3Error.songsOnly
    }

    public func generateAudio(
        _ request: FamilyRequest, phase: (AudioPhase) -> Void, progress: (Int, Int) -> Void, isCancelled: () -> Bool
    ) throws -> GeneratedAudio {
        try Music3Error.validate(request)
        let duration = request.duration ?? Self.defaultDuration
        let maxFrames = min(max(1, Int(duration * config.frameRate)), Music3Config.maxAudioFrames)

        phase(.encoding)
        let ids = try promptIDs(caption: request.prompt, lyrics: request.lyrics ?? "")
        phase(.composing(frames: maxFrames))
        let frames = try self.frames(ids: ids, maxFrames: maxFrames, seed: request.seed,
                                     progress: { progress($0, maxFrames) }, isCancelled: isCancelled)

        let total = Music3Flow.chunkStarts(frames: frames.dim(1)).count * request.steps
        phase(.denoising(total: total))
        let chunks = try latents(frames: frames, steps: request.steps, seed: request.seed, progress: progress, isCancelled: isCancelled)

        phase(.decoding)
        let wave = try decode(chunks, isCancelled: isCancelled)
        let samples = clip(wave[0].transposed(1, 0).asType(.float32), min: -1, max: 1)
        eval(samples)
        return GeneratedAudio(samples: samples, sampleRate: config.sampleRate, frames: frames.dim(1))
    }
}
