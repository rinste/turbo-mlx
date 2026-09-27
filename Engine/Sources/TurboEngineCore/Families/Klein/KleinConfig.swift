import Foundation

/// The geometry of a FLUX.2 Klein checkpoint. Mirrors `flux2-klein-*` in mflux's model registry;
/// a fixture (see Fixtures/) may shrink every dimension for a parity check.
public struct KleinConfig: Equatable, Sendable {
    public struct Transformer: Equatable, Sendable {
        public var numLayers: Int
        public var numSingleLayers: Int
        public var numAttentionHeads: Int
        public var jointAttentionDim: Int
        public var attentionHeadDim = 128
        public var inChannels = 128
        public var mlpRatio: Double = 3.0
        public var timestepGuidanceChannels = 256
        public var ropeAxesDim = [32, 32, 32, 32]
        public var ropeTheta: Double = 2000

        public var innerDim: Int { numAttentionHeads * attentionHeadDim }
    }

    public struct TextEncoder: Equatable, Sendable {
        public var vocabSize = 151_936
        public var hiddenSize: Int
        public var numHiddenLayers = 36
        public var numAttentionHeads = 32
        public var numKeyValueHeads = 8
        public var intermediateSize: Int
        public var headDim = 128
        public var ropeTheta: Double = 1_000_000
        public var rmsNormEps: Float = 1e-6
        /// The epsilon of the per-head query and key norms: Klein's encoder uses the layer norm's
        /// (1e-6), Z-Image's the MLX default (1e-5).
        public var qkNormEps: Float = 1e-6

        public init(hiddenSize: Int, intermediateSize: Int, qkNormEps: Float = 1e-6) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.qkNormEps = qkNormEps
        }
    }

    public var name: String
    public var transformer: Transformer
    public var textEncoder: TextEncoder
    /// Hidden states of these layers (embeddings being layer 0) are concatenated into the prompt.
    public var textEncoderOutLayers = [9, 18, 27]
    public var maxSequenceLength = 512
    public var defaultSteps = 4
    /// A base (not distilled) checkpoint: many steps and real classifier-free guidance.
    public var isBase = false

    public static let klein4B = KleinConfig(
        name: "flux2-klein-4b",
        transformer: Transformer(numLayers: 5, numSingleLayers: 20, numAttentionHeads: 24, jointAttentionDim: 7680),
        textEncoder: TextEncoder(hiddenSize: 2560, intermediateSize: 9728)
    )

    public static let klein9B = KleinConfig(
        name: "flux2-klein-9b",
        transformer: Transformer(numLayers: 8, numSingleLayers: 24, numAttentionHeads: 32, jointAttentionDim: 12288),
        textEncoder: TextEncoder(hiddenSize: 4096, intermediateSize: 12288)
    )

    /// The 4B geometry unless the model's registry key or name says 9B, as mflux infers it; a
    /// "base" in either marks the undistilled checkpoint.
    public static func forModel(name: String?, variant: String?) -> KleinConfig {
        let hints = [variant ?? "", name ?? ""].joined(separator: " ").lowercased()
        var config = hints.contains("9b") ? klein9B : klein4B
        config.isBase = hints.contains("base")
        if config.isBase { config.defaultSteps = 50 }
        return config
    }

    /// The geometry a fixture's `fixture.json` describes (see Fixtures/make_klein_fixture.py).
    public static func fixture(_ json: [String: Any]) -> KleinConfig {
        let t = json["transformer"] as? [String: Any] ?? [:]
        let e = json["text_encoder"] as? [String: Any] ?? [:]
        func int(_ dict: [String: Any], _ key: String, _ fallback: Int) -> Int { (dict[key] as? NSNumber)?.intValue ?? fallback }
        var transformer = Transformer(
            numLayers: int(t, "num_layers", 5),
            numSingleLayers: int(t, "num_single_layers", 20),
            numAttentionHeads: int(t, "num_attention_heads", 24),
            jointAttentionDim: int(t, "joint_attention_dim", 7680)
        )
        transformer.attentionHeadDim = int(t, "attention_head_dim", 128)
        var encoder = TextEncoder(hiddenSize: int(e, "hidden_size", 2560), intermediateSize: int(e, "intermediate_size", 9728))
        encoder.vocabSize = int(e, "vocab_size", 151_936)
        encoder.numHiddenLayers = int(e, "num_hidden_layers", 36)
        encoder.numAttentionHeads = int(e, "num_attention_heads", 32)
        encoder.numKeyValueHeads = int(e, "num_key_value_heads", 8)
        encoder.headDim = int(e, "head_dim", 128)
        var config = KleinConfig(name: json["variant"] as? String ?? "fixture", transformer: transformer, textEncoder: encoder)
        if let layers = json["text_encoder_out_layers"] as? [NSNumber] { config.textEncoderOutLayers = layers.map(\.intValue) }
        config.maxSequenceLength = int(json, "max_sequence_length", 512)
        return config
    }
}
