import Foundation

/// The geometry of a Qwen-Image checkpoint (mflux's `qwen-image`, Qwen-Image 2512); a fixture may
/// shrink it.
public struct QwenImageConfig: Equatable, Sendable {
    public struct Transformer: Equatable, Sendable {
        public var numLayers = 60
        public var numAttentionHeads = 24
        public var attentionHeadDim = 128
        public var jointAttentionDim = 3584
        public var inChannels = 64
        public var outChannels = 16
        public var ropeTheta: Float = 10000
        public var ropeAxesDim = [16, 56, 56]

        public var innerDim: Int { numAttentionHeads * attentionHeadDim }
        public static let patchSize = 2

        public init() {}
    }

    /// Qwen2.5-VL 7B's language model, read after its final norm, in bf16.
    public struct TextEncoder: Equatable, Sendable {
        public var vocabSize = 152_064
        public var hiddenSize = 3584
        public var numHiddenLayers = 28
        public var numAttentionHeads = 28
        public var numKeyValueHeads = 4
        public var intermediateSize = 18944
        public var ropeTheta: Double = 1_000_000
        public var rmsNormEps: Float = 1e-6
        /// The template's system prompt: that many leading tokens are dropped from the output.
        public var dropIndex = 34

        public var headDim: Int { hiddenSize / numAttentionHeads }

        public init() {}
    }

    public var name: String
    public var transformer: Transformer
    public var textEncoder: TextEncoder
    /// Decoder width of the first stage (96 in the model; the stages are 1×, 2×, 4× of it).
    public var vaeBaseDim = 96
    public var maxSequenceLength = 1058
    public var defaultSteps = 20
    /// `requires_sigma_shift` as the model config sets it: 0.5 → 0.9 over 256 → 8192 tokens, with
    /// the terminal sigma stretched to 0.02.
    public var shift = LinearSchedule.Shift(baseShift: 0.5, maxShift: 0.9, baseSeqLen: 256, maxSeqLen: 8192, terminal: 0.02)

    /// mflux's tokenizer template around the prompt (`{}`).
    public static let template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    /// Qwen's own pipeline encodes an empty negative prompt as a single space.
    public static let negativePrompt = " "

    public static let qwenImage2512 = QwenImageConfig(name: "qwen-image", transformer: Transformer(), textEncoder: TextEncoder())

    /// The geometry a fixture's `fixture.json` describes (see Fixtures/make_qwen_image_fixture.py).
    public static func fixture(_ json: [String: Any]) -> QwenImageConfig {
        let t = json["transformer"] as? [String: Any] ?? [:]
        let e = json["text_encoder"] as? [String: Any] ?? [:]
        let v = json["vae"] as? [String: Any] ?? [:]
        func int(_ dict: [String: Any], _ key: String, _ fallback: Int) -> Int { (dict[key] as? NSNumber)?.intValue ?? fallback }
        var transformer = Transformer()
        transformer.numLayers = int(t, "num_layers", transformer.numLayers)
        transformer.numAttentionHeads = int(t, "num_attention_heads", transformer.numAttentionHeads)
        transformer.attentionHeadDim = int(t, "attention_head_dim", transformer.attentionHeadDim)
        transformer.jointAttentionDim = int(t, "joint_attention_dim", transformer.jointAttentionDim)
        var encoder = TextEncoder()
        encoder.vocabSize = int(e, "vocab_size", encoder.vocabSize)
        encoder.hiddenSize = int(e, "hidden_size", encoder.hiddenSize)
        encoder.numHiddenLayers = int(e, "num_hidden_layers", encoder.numHiddenLayers)
        encoder.numAttentionHeads = int(e, "num_attention_heads", encoder.numAttentionHeads)
        encoder.numKeyValueHeads = int(e, "num_key_value_heads", encoder.numKeyValueHeads)
        encoder.intermediateSize = int(e, "intermediate_size", encoder.intermediateSize)
        var config = QwenImageConfig(name: json["variant"] as? String ?? "fixture", transformer: transformer, textEncoder: encoder)
        config.vaeBaseDim = int(v, "base_dim", config.vaeBaseDim)
        return config
    }
}
