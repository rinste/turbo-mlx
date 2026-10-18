import Foundation

/// The geometry of a FLUX 3 Image Klein checkpoint. Mirrors `flux3-image-klein*` in mflux's model
/// registry and `Flux3Transformer`'s defaults; a fixture (see Fixtures/) may shrink every dimension
/// for a parity check.
public struct Flux3Config: Equatable, Sendable {
    public struct Transformer: Equatable, Sendable {
        /// Per-mode blocks (each of image, references and text has its own), then single-stream ones.
        public var numLayers = 5
        public var numSingleLayers = 28
        public var numAttentionHeads = 24
        public var attentionHeadDim = 128
        public var jointAttentionDim = 20480
        public var inChannels = 128
        public var timestepChannels = 256
        public var mlpRatio: Double = 3.0
        public var ropeAxesDim = [32, 32, 32, 32]
        public var ropeTheta: Double = 10000
        /// The distilled checkpoint is a flow map: a second time embedding gives the step's target.
        public var flowMap = true

        public var innerDim: Int { numAttentionHeads * attentionHeadDim }
    }

    public var name: String
    public var transformer: Transformer
    /// The language tower of Qwen3-VL-4B, read up to layer 32 of its 36.
    public var textEncoder: KleinConfig.TextEncoder
    /// Hidden states of these layers (embeddings being layer 0) are concatenated into the prompt.
    public var textEncoderOutLayers = [4, 8, 12, 16, 20, 24, 28, 32]
    /// The prompt's tokens are padded to a multiple of `padMultiple`, at most `maxSequenceLength`.
    public var maxSequenceLength = 4096
    public var padMultiple = 80
    public var defaultSteps = 16
    /// The undistilled Base checkpoint: 50 steps and classifier-free guidance against an empty prompt.
    public var isBase: Bool { !transformer.flowMap }

    static func qwen3VL4B() -> KleinConfig.TextEncoder {
        var encoder = KleinConfig.TextEncoder(hiddenSize: 2560, intermediateSize: 9728)
        encoder.numHiddenLayers = 32
        encoder.ropeTheta = 5_000_000
        return encoder
    }

    public static let klein = Flux3Config(name: "flux3-image-klein", transformer: Transformer(), textEncoder: qwen3VL4B())

    public static let kleinBase: Flux3Config = {
        var config = Flux3Config(name: "flux3-image-klein-base", transformer: Transformer(), textEncoder: qwen3VL4B())
        config.transformer.flowMap = false
        config.defaultSteps = 50
        return config
    }()

    /// The distilled checkpoint unless the model's registry key or name says "base", as mflux's
    /// `--base-model` picks it.
    public static func forModel(name: String?, variant: String?) -> Flux3Config {
        let hints = [variant ?? "", name ?? ""].joined(separator: " ").lowercased()
        return hints.contains("base") ? kleinBase : klein
    }

    /// The geometry a fixture's `fixture.json` describes (see Fixtures/make_flux3_fixture.py).
    public static func fixture(_ json: [String: Any]) -> Flux3Config {
        let t = json["transformer"] as? [String: Any] ?? [:]
        let e = json["text_encoder"] as? [String: Any] ?? [:]
        func int(_ dict: [String: Any], _ key: String, _ fallback: Int) -> Int { (dict[key] as? NSNumber)?.intValue ?? fallback }
        var transformer = Transformer()
        transformer.numLayers = int(t, "num_layers", transformer.numLayers)
        transformer.numSingleLayers = int(t, "num_single_layers", transformer.numSingleLayers)
        transformer.numAttentionHeads = int(t, "num_attention_heads", transformer.numAttentionHeads)
        transformer.attentionHeadDim = int(t, "attention_head_dim", transformer.attentionHeadDim)
        transformer.jointAttentionDim = int(t, "joint_attention_dim", transformer.jointAttentionDim)
        transformer.flowMap = (t["flow_map"] as? Bool) ?? true
        var encoder = KleinConfig.TextEncoder(hiddenSize: int(e, "hidden_size", 2560), intermediateSize: int(e, "intermediate_size", 9728))
        encoder.vocabSize = int(e, "vocab_size", 151_936)
        encoder.numHiddenLayers = int(e, "num_hidden_layers", 32)
        encoder.numAttentionHeads = int(e, "num_attention_heads", 32)
        encoder.numKeyValueHeads = int(e, "num_key_value_heads", 8)
        encoder.headDim = int(e, "head_dim", 128)
        encoder.ropeTheta = (e["rope_theta"] as? NSNumber)?.doubleValue ?? 5_000_000
        var config = Flux3Config(name: json["variant"] as? String ?? "fixture", transformer: transformer, textEncoder: encoder)
        if let layers = json["text_encoder_out_layers"] as? [NSNumber] { config.textEncoderOutLayers = layers.map(\.intValue) }
        config.maxSequenceLength = int(json, "max_sequence_length", config.maxSequenceLength)
        config.padMultiple = int(json, "pad_multiple", config.padMultiple)
        config.defaultSteps = config.isBase ? 50 : 16
        return config
    }
}
