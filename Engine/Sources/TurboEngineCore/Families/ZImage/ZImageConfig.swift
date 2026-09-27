import Foundation

/// The geometry of a Z-Image Turbo checkpoint (mflux's `z-image-turbo`); a fixture may shrink it.
public struct ZImageConfig: Equatable, Sendable {
    public var name: String
    public var transformer: S3DiTConfig
    /// Qwen3 4B, read for its second-to-last hidden state, in float32.
    public var textEncoder: KleinConfig.TextEncoder
    /// Channels of the decoder's four stages, first stage first.
    public var vaeBlockOutChannels = [128, 256, 512, 512]
    public var maxSequenceLength = 512
    public var defaultSteps = 9
    /// `requires_sigma_shift` with the mflux defaults (0.5 → 1.15 over 256 → 4096 tokens).
    public var shift = LinearSchedule.Shift()

    public static let turbo = ZImageConfig(
        name: "z-image-turbo",
        transformer: S3DiTConfig(),
        textEncoder: KleinConfig.TextEncoder(hiddenSize: 2560, intermediateSize: 9728, qkNormEps: 1e-5)
    )

    /// The geometry a fixture's `fixture.json` describes (see Fixtures/make_zimage_fixture.py).
    public static func fixture(_ json: [String: Any]) -> ZImageConfig {
        let t = json["transformer"] as? [String: Any] ?? [:]
        let e = json["text_encoder"] as? [String: Any] ?? [:]
        let v = json["vae"] as? [String: Any] ?? [:]
        func int(_ dict: [String: Any], _ key: String, _ fallback: Int) -> Int { (dict[key] as? NSNumber)?.intValue ?? fallback }
        var transformer = S3DiTConfig()
        transformer.dim = int(t, "dim", transformer.dim)
        transformer.numLayers = int(t, "n_layers", transformer.numLayers)
        transformer.numRefinerLayers = int(t, "n_refiner_layers", transformer.numRefinerLayers)
        transformer.numHeads = int(t, "n_heads", transformer.numHeads)
        transformer.capFeatDim = int(t, "cap_feat_dim", transformer.capFeatDim)
        if let dims = t["axes_dims"] as? [NSNumber] { transformer.axesDims = dims.map(\.intValue) }
        if let lens = t["axes_lens"] as? [NSNumber] { transformer.axesLens = lens.map(\.intValue) }
        var encoder = KleinConfig.TextEncoder(hiddenSize: int(e, "hidden_size", 2560), intermediateSize: int(e, "intermediate_size", 9728), qkNormEps: 1e-5)
        encoder.vocabSize = int(e, "vocab_size", 151_936)
        encoder.numHiddenLayers = int(e, "num_hidden_layers", 36)
        encoder.numAttentionHeads = int(e, "num_attention_heads", 32)
        encoder.numKeyValueHeads = int(e, "num_key_value_heads", 8)
        encoder.headDim = int(e, "head_dim", 128)
        var config = ZImageConfig(name: json["variant"] as? String ?? "fixture", transformer: transformer, textEncoder: encoder)
        if let channels = v["block_out_channels"] as? [NSNumber] { config.vaeBlockOutChannels = channels.map(\.intValue) }
        config.maxSequenceLength = int(json, "max_sequence_length", 512)
        return config
    }
}
