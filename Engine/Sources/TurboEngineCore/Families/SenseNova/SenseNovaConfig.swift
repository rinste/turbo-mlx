import Foundation

/// The geometry of a SenseNova-U1.5 checkpoint, read from the `config.json` next to its weights
/// (the MLX packs keep the original's, `NEOChatConfig`): the 8B-MoT model is two Qwen3 stacks of
/// 42 layers of 4096 (one reads the prompt, the other the image), a 16-pixel patch embedding merged
/// 2 × 2 into tokens of 32 pixels, and a convolutional pixel head. A fixture shrinks it.
public struct SenseNovaConfig: Equatable, Sendable {
    public var hiddenSize = 4096
    public var numLayers = 42
    public var numAttentionHeads = 32
    public var numKeyValueHeads = 8
    public var headDim = 128
    public var intermediateSize = 12288
    public var vocabSize = 151_936
    public var rmsNormEps: Float = 1e-6
    /// The text-position rotary base, and the one of rows and columns.
    public var ropeTheta: Double = 5_000_000
    public var ropeThetaHW: Double = 10_000
    /// The patch embedding: 16-pixel patches into `visionHiddenSize` features, then 2 × 2 of them
    /// into one token; its rotary base.
    public var patchSize = 16
    public var mergeSize = 2
    public var visionHiddenSize = 1024
    public var visionRopeTheta: Double = 10_000
    /// The noise's scale grows with the square root of the token count over this base, up to the
    /// maximum; the model is told it (divided by the maximum) through its own embedder.
    public var noiseScale: Double = 1
    public var noiseScaleBaseSeqLen = 64
    public var noiseScaleMaxValue: Double = 16
    public var addNoiseScaleEmbedding = true
    /// The quantization the pack was saved with (nil: unquantized).
    public var bits: Int?

    /// Pixels per image token.
    public var tokenSize: Int { patchSize * mergeSize }

    public init() {}

    public enum ConfigError: LocalizedError {
        case unreadable(URL)
        case unsupported(String)

        public var errorDescription: String? {
            switch self {
            case .unreadable(let url): "Couldn’t read the model’s configuration at \(url.path)."
            case .unsupported(let what): "This SenseNova checkpoint uses \(what), which the engine does not run."
            }
        }
    }

    /// Reads `config.json` in `folder`; the settings the port does not implement are refused
    /// rather than ignored.
    public static func read(folder: URL) throws -> SenseNovaConfig {
        let url = folder.appending(path: "config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let llm = json["llm_config"] as? [String: Any]
        else { throw ConfigError.unreadable(url) }
        func number(_ object: [String: Any], _ key: String) -> Double? { (object[key] as? NSNumber)?.doubleValue }

        if (llm["model_type"] as? String).map({ $0 != "qwen3" }) ?? false { throw ConfigError.unsupported("a mixture-of-experts backbone") }
        if (json["use_pixel_head"] as? Bool) == false { throw ConfigError.unsupported("a flow-matching head without the pixel decoder") }
        if (number(json, "concat_time_token_num") ?? 0) != 0 { throw ConfigError.unsupported("time tokens") }
        if let mode = json["noise_scale_mode"] as? String, mode != "resolution" { throw ConfigError.unsupported("the noise scale mode \(mode)") }

        var config = SenseNovaConfig()
        config.hiddenSize = number(llm, "hidden_size").map(Int.init) ?? config.hiddenSize
        config.numLayers = number(llm, "num_hidden_layers").map(Int.init) ?? config.numLayers
        config.numAttentionHeads = number(llm, "num_attention_heads").map(Int.init) ?? config.numAttentionHeads
        config.numKeyValueHeads = number(llm, "num_key_value_heads").map(Int.init) ?? config.numKeyValueHeads
        config.headDim = number(llm, "head_dim").map(Int.init) ?? config.hiddenSize / config.numAttentionHeads
        config.intermediateSize = number(llm, "intermediate_size").map(Int.init) ?? config.intermediateSize
        config.vocabSize = number(llm, "vocab_size").map(Int.init) ?? config.vocabSize
        config.rmsNormEps = number(llm, "rms_norm_eps").map(Float.init) ?? config.rmsNormEps
        config.ropeTheta = number(llm, "rope_theta") ?? config.ropeTheta
        config.ropeThetaHW = number(llm, "rope_theta_hw") ?? config.ropeThetaHW
        if let vision = json["vision_config"] as? [String: Any] {
            config.visionHiddenSize = number(vision, "hidden_size").map(Int.init) ?? config.visionHiddenSize
            config.visionRopeTheta = number(vision, "rope_theta_vision") ?? config.visionRopeTheta
            config.patchSize = number(vision, "patch_size").map(Int.init) ?? config.patchSize
        }
        if let ratio = number(json, "downsample_ratio"), ratio > 0 { config.mergeSize = Int((1 / ratio).rounded()) }
        config.noiseScale = number(json, "noise_scale") ?? config.noiseScale
        config.noiseScaleBaseSeqLen = number(json, "noise_scale_base_image_seq_len").map(Int.init) ?? config.noiseScaleBaseSeqLen
        config.noiseScaleMaxValue = number(json, "noise_scale_max_value") ?? config.noiseScaleMaxValue
        config.addNoiseScaleEmbedding = json["add_noise_scale_embedding"] as? Bool ?? config.addNoiseScaleEmbedding
        if let quantization = json["quantization"] as? [String: Any] { config.bits = number(quantization, "bits").map(Int.init) }
        guard config.headDim % 4 == 0 else { throw ConfigError.unsupported("a head size of \(config.headDim)") }
        return config
    }

    /// The noise's standard deviation for an image of `tokens` tokens (`noise_scale_mode` "resolution").
    public func noiseScale(tokens: Int) -> Double {
        min((Double(tokens) / Double(noiseScaleBaseSeqLen)).squareRoot() * noiseScale, noiseScaleMaxValue)
    }

    /// `_apply_time_schedule` with the "standard" schedule: t from 0 (noise) to 1 (the image),
    /// `steps + 1` values, shifted towards the noise.
    public static func timesteps(steps: Int, shift: Double) -> [Float] {
        (0 ... steps).map { index in
            let t = Float(index) / Float(steps)
            let sigma = 1 - t
            let shifted = Float(shift) * sigma / (1 + (Float(shift) - 1) * sigma)
            return 1 - shifted
        }
    }

    /// The system message every generation prompt starts with (`SYSTEM_MESSAGE_FOR_GEN`).
    public static let systemMessage = """
        You are an image generation and editing assistant that accurately understands and executes user intent.

        You support two modes:

        1. Think Mode:
        If the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.

        2. Non-Think Mode:
        If no reasoning is needed, directly produce the final image.

        Task Types:

        A. Text-to-Image Generation:
        - Generate a high-quality image based on the user's description.
        - Ensure visual clarity, semantic consistency, and completeness.
        - DO NOT introduce elements that contradict or override the user's intent.

        B. Image Editing:
        - Use the provided image(s) as input or reference for modification or transformation.
        - The result can be an edited image or a new image based on the reference(s).
        - Preserve all unspecified attributes unless explicitly changed.

        General Rules:
        - For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. If no language is specified, use the user's input language.
        """

    /// The "neo1_0" chat template around a generation prompt, thinking off, ending where the image
    /// starts (`_build_t2i_query` with `SYSTEM_MESSAGE_FOR_GEN` and the empty think block).
    public static func conditionalQuery(_ prompt: String) -> String {
        "<|im_start|>system\n\(systemMessage)<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
    }

    /// The unconditional query of classifier-free guidance: no system message, an empty turn.
    public static let unconditionalQuery = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>"
}
