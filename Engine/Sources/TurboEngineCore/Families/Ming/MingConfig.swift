import Foundation

/// The geometry of a Ming-Image 0.1 Design checkpoint (mflux's `ming-image-design`); a fixture may
/// shrink every part of it.
public struct MingConfig: Equatable, Sendable {
    /// Ling-mini-2.0 (`bailing_moe_v2`) as shipped in the checkpoint's `mllm/` folder: a
    /// mixture-of-experts decoder read for its hidden states.
    public struct Encoder: Equatable, Sendable {
        public var hiddenSize = 2048
        public var numLayers = 20
        public var numHeads = 16
        public var numKvHeads = 4
        public var headDim = 128
        /// `partial_rotary_factor` 0.5: the first 64 of each head's 128 dims are rotated.
        public var ropeDim = 64
        public var ropeTheta: Float = 600_000
        public var rmsEps: Float = 1e-6
        public var vocabSize = 157_184
        public var denseIntermediate = 5120
        /// The first layer is a dense MLP; the others route to experts.
        public var firstKDense = 1
        public var numExperts = 256
        public var topK = 8
        public var nGroup = 8
        public var topkGroup = 4
        public var moeIntermediate = 512
        public var routedScaling: Float = 2.5

        public init() {}
    }

    /// The `connector/` folder: a Qwen2-1.5B decoder stack used bidirectionally over the query tokens.
    public struct Connector: Equatable, Sendable {
        public var hiddenSize = 1536
        public var numLayers = 28
        public var numHeads = 12
        public var numKvHeads = 2
        public var headDim = 128
        public var intermediate = 8960
        public var ropeTheta: Float = 1_000_000
        public var rmsEps: Float = 1e-6

        public init() {}
    }

    /// The `mlp/` folder: the learned query tokens and the projections around the connector and
    /// the direct-VLM head.
    public struct Heads: Equatable, Sendable {
        /// `img_gen_scales = [16]`: a 16 × 16 grid of query tokens.
        public var queryGrid = 16
        /// The encoder's hidden states (input of layer i; the count means after the final norm)
        /// concatenated into the direct-VLM tokens.
        public var directVLMLayers = [5, 12, 20]
        public var capFeatDim = 2560
        public var ditDim = 3840

        public var queryCount: Int { queryGrid * queryGrid }

        public init() {}
    }

    public var name: String
    public var encoder: Encoder
    public var connector: Connector
    public var heads: Heads
    public var transformer: S3DiTConfig
    public var vaeBaseDim = 96
    public var imageStartId = 157_158
    public var imagePatchId = 157_157
    public var imageEndId = 157_159
    public var defaultSteps = 12
    /// The checkpoint's static shift (its dynamic-shifting override never reaches the frozen config).
    public var sigmaShift: Float = 6
    /// Latents are scaled by a single factor for this VAE.
    public var vaeScalingFactor: Float = 8.0064

    /// Rendered by the checkpoint's chat template for a single HUMAN turn with generation prompt.
    public static let promptTemplate = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>{}<|role_end|><role>ASSISTANT</role>"

    /// The S3-DiT as Ming runs it: no pad tokens, activations in the weights' dtype, and a rotary
    /// t axis long enough for the caption offset.
    static func transformerDefaults() -> S3DiTConfig {
        var config = S3DiTConfig()
        config.axesLens = [20480, 512, 512]
        config.padsToMultiple = false
        config.keepsWeightsPrecision = true
        return config
    }

    public static let design = MingConfig(
        name: "ming-image-design",
        encoder: Encoder(),
        connector: Connector(),
        heads: Heads(),
        transformer: transformerDefaults()
    )

    /// The geometry a fixture's `fixture.json` describes (see Fixtures/make_ming_fixture.py).
    public static func fixture(_ json: [String: Any]) -> MingConfig {
        func int(_ dict: [String: Any], _ key: String, _ fallback: Int) -> Int { (dict[key] as? NSNumber)?.intValue ?? fallback }
        let e = json["encoder"] as? [String: Any] ?? [:]
        let c = json["connector"] as? [String: Any] ?? [:]
        let h = json["heads"] as? [String: Any] ?? [:]
        let t = json["transformer"] as? [String: Any] ?? [:]
        let v = json["vae"] as? [String: Any] ?? [:]

        var encoder = Encoder()
        encoder.hiddenSize = int(e, "hidden_size", encoder.hiddenSize)
        encoder.numLayers = int(e, "num_layers", encoder.numLayers)
        encoder.numHeads = int(e, "num_heads", encoder.numHeads)
        encoder.numKvHeads = int(e, "num_kv_heads", encoder.numKvHeads)
        encoder.vocabSize = int(e, "vocab_size", encoder.vocabSize)
        encoder.denseIntermediate = int(e, "dense_intermediate", encoder.denseIntermediate)
        encoder.numExperts = int(e, "num_experts", encoder.numExperts)
        encoder.topK = int(e, "top_k", encoder.topK)
        encoder.nGroup = int(e, "n_group", encoder.nGroup)
        encoder.topkGroup = int(e, "topk_group", encoder.topkGroup)
        encoder.moeIntermediate = int(e, "moe_intermediate", encoder.moeIntermediate)

        var connector = Connector()
        connector.hiddenSize = int(c, "hidden_size", connector.hiddenSize)
        connector.numLayers = int(c, "num_layers", connector.numLayers)
        connector.numHeads = int(c, "num_heads", connector.numHeads)
        connector.numKvHeads = int(c, "num_kv_heads", connector.numKvHeads)
        connector.intermediate = int(c, "intermediate", connector.intermediate)

        var heads = Heads()
        heads.queryGrid = int(h, "query_grid", heads.queryGrid)
        if let layers = h["directvlm_layers"] as? [NSNumber] { heads.directVLMLayers = layers.map(\.intValue) }
        heads.capFeatDim = int(h, "cap_feat_dim", heads.capFeatDim)
        heads.ditDim = int(h, "dit_dim", heads.ditDim)

        var transformer = transformerDefaults()
        transformer.dim = int(t, "dim", transformer.dim)
        transformer.numLayers = int(t, "n_layers", transformer.numLayers)
        transformer.numRefinerLayers = int(t, "n_refiner_layers", transformer.numRefinerLayers)
        transformer.numHeads = int(t, "n_heads", transformer.numHeads)
        transformer.capFeatDim = int(t, "cap_feat_dim", transformer.capFeatDim)

        var config = MingConfig(name: json["variant"] as? String ?? "fixture", encoder: encoder, connector: connector, heads: heads, transformer: transformer)
        config.vaeBaseDim = int(v, "base_dim", config.vaeBaseDim)
        config.imageStartId = int(json, "image_start_id", config.imageStartId)
        config.imagePatchId = int(json, "image_patch_id", config.imagePatchId)
        config.imageEndId = int(json, "image_end_id", config.imageEndId)
        return config
    }
}
