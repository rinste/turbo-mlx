import Foundation

/// MiniMax Music 3's sizes and the checkpoint's token contract, read from the pack's `config.json` (mlx-audio's
/// `ModelConfig`, snake_case keys; a key the file lacks keeps the official model's value). The constants below are the
/// reference's own (`mlx_audio/music/models/minimax_music3/config.py`).
public struct Music3Config {
    public var hiddenSize = 4096
    public var numCodebooks = 8
    public var audioVocabSize = 1024
    public var semanticVocabSize = 16384
    public var vocabSize = 200_000
    public var audioCodeOffset = 151_675
    public var audioEndTokenID = 151_670
    public var audioCFGTokenID = 151_654

    public var numHiddenLayers = 36
    public var intermediateSize = 12288
    public var numAttentionHeads = 32
    public var numKeyValueHeads = 8
    public var headDim = 128
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 1_000_000
    public var tieWordEmbeddings = false

    public var depthNumLayers = 4
    public var depthNumHeads = 16
    public var depthIntermediateSize = 6144
    public var depthMaxPositionEmbeddings = 16

    public var conditionOutDim = 2048
    public var numConditionLayers = 8
    public var inputSamplingRate = 24000
    public var inputHopLength = 960
    public var outputSamplingRate = 44100
    public var outputHopLength = 512

    public var ditInChannels = 128
    public var ditNumLayers = 36
    public var ditNumHeads = 32
    public var ditHeadDim = 64
    public var ditFFInnerDim = 8192
    public var ditRotaryDim = 32
    public var ditFourierDim = 256

    public var vocoderInputDim = 1024
    public var vocoderHiddenDim = 1536
    public var vocoderUpsamplingRatios = [8, 8, 4, 2]
    public var sampleRate = 44100
    public var frameRate = 25.0

    /// Classifier-free guidance of the global and local language models, and the top-k both sample from.
    public static let arGuidance: Float = 1.5
    public static let arTopK = 50
    /// The flow transformer's guidance.
    public static let ditGuidance: Float = 1.7
    public static let maxAudioFrames = 9000
    public static let maxPromptTokens = 5000
    /// The longest song the reference generates, in seconds, and the most flow steps it takes.
    public static let maxDuration = 360.0
    public static let maxSteps = 30
    /// The flow runs on windows of 200 frames (8 s) every 100 frames; each carries 172 latents of the one before it and
    /// keeps the middle of its waveform: 86 latents cut on the left, 258 on the right (none at the song's ends).
    public static let chunkFrames = 200
    public static let chunkHop = 100
    public static let overlapLatents = 172
    public static let cropLeftLatents = 86
    public static let cropRightLatents = 344 - 86

    public var residualCodebooks: Int { numCodebooks - 1 }

    public init() {}

    public static func read(folder: URL) throws -> Music3Config {
        let url = folder.appending(path: "config.json")
        guard let data = try? Data(contentsOf: url), let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw Music3Error.missingFile("config.json", folder) }
        return Music3Config(json: json)
    }

    public init(json: [String: Any]) {
        func int(_ key: String, _ value: inout Int) { if let number = json[key] as? NSNumber { value = number.intValue } }
        func float(_ key: String, _ value: inout Float) { if let number = json[key] as? NSNumber { value = number.floatValue } }
        int("hidden_size", &hiddenSize)
        int("num_codebooks", &numCodebooks)
        int("audio_vocab_size", &audioVocabSize)
        int("semantic_vocab_size", &semanticVocabSize)
        int("vocab_size", &vocabSize)
        int("audio_code_offset", &audioCodeOffset)
        int("audio_end_token_id", &audioEndTokenID)
        int("audio_cfg_token_id", &audioCFGTokenID)
        int("num_hidden_layers", &numHiddenLayers)
        int("intermediate_size", &intermediateSize)
        int("num_attention_heads", &numAttentionHeads)
        int("num_key_value_heads", &numKeyValueHeads)
        int("head_dim", &headDim)
        float("rms_norm_eps", &rmsNormEps)
        float("rope_theta", &ropeTheta)
        if let tie = json["tie_word_embeddings"] as? Bool { tieWordEmbeddings = tie }
        int("depth_num_layers", &depthNumLayers)
        int("depth_num_heads", &depthNumHeads)
        int("depth_intermediate_size", &depthIntermediateSize)
        int("depth_max_position_embeddings", &depthMaxPositionEmbeddings)
        int("condition_out_dim", &conditionOutDim)
        int("num_condition_layers", &numConditionLayers)
        int("input_sampling_rate", &inputSamplingRate)
        int("input_hop_length", &inputHopLength)
        int("output_sampling_rate", &outputSamplingRate)
        int("output_hop_length", &outputHopLength)
        int("dit_in_channels", &ditInChannels)
        int("dit_num_layers", &ditNumLayers)
        int("dit_num_heads", &ditNumHeads)
        int("dit_head_dim", &ditHeadDim)
        int("dit_ff_inner_dim", &ditFFInnerDim)
        int("dit_rotary_dim", &ditRotaryDim)
        int("dit_fourier_dim", &ditFourierDim)
        int("vocoder_input_dim", &vocoderInputDim)
        int("vocoder_hidden_dim", &vocoderHiddenDim)
        if let ratios = json["vocoder_upsampling_ratios"] as? [NSNumber] { vocoderUpsamplingRatios = ratios.map(\.intValue) }
        int("sample_rate", &sampleRate)
        if let rate = json["frame_rate"] as? NSNumber { frameRate = rate.doubleValue }
    }

    /// The flow's latents for `frames` language-model frames: `latent_length_from_frames`.
    func latentLength(frames: Int) -> Int {
        max(1, Int(Double(frames) * Double(outputSamplingRate) / Double(inputSamplingRate) * Double(inputHopLength) / Double(outputHopLength)))
    }
}
