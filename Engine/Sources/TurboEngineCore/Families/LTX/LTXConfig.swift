import Foundation

// LTX-2.3, as dgrauet's MLX port runs it (`ltx-2-mlx`, the reference this family is checked
// against): the shapes of the pack's `embedded_config.json` and the tables of its distilled
// two-stage pipeline (`ltx_pipelines_mlx/distilled.py`). The pack is one of dgrauet's
// `ltx-2.3-mlx-q4` / `-q8` folders (a safetensors file per component, keyed by the port's module
// tree); the text encoder, Gemma 3 12B, is `mlx-community/gemma-3-12b-it-4bit`, a folder of its own.

public struct LTXConfig {
    /// `LTXModelConfig.from_checkpoint_config`: the DiT's shapes and its timestep and RoPE settings.
    public struct Transformer {
        public var numLayers = 48
        public var videoDim = 4096
        public var audioDim = 2048
        public var videoHeads = 32
        public var audioHeads = 32
        public var videoHeadDim = 128
        public var audioHeadDim = 64
        public var crossHeads = 32
        public var crossHeadDim = 64
        public var videoPatchChannels = 128
        public var audioPatchChannels = 128
        public var ffMult = 4
        public var timestepEmbeddingDim = 256
        public var timestepScale: Float = 1000
        /// The audio–video cross-attention gates' timestep scale (1000 in every 2.3 checkpoint).
        public var crossTimestepScale: Float = 1000
        public var ropeTheta: Double = 10000
        public var maxPositions = [20, 2048, 2048]
        public var audioMaxPositions = [20]
        public var normEps: Float = 1e-6
        public var ffBias = true
        public var audioFFBias = true
        /// `frequencies_precision: float64`: the RoPE frequency grid is computed in float64.
        public var doublePrecisionRope = true
        /// LTX-2.5: `use_keyframes_abs_pos_embedding`, a learned marker on the first latent frame.
        public var keyframesEmbedding = false
    }

    /// `GemmaFeaturesExtractorV2` and its two `Embeddings1DConnector`s (fixed in the port).
    public struct Connector {
        public var gemmaLayers = 49
        public var captionChannels = 3840
        public var heads = 32
        public var videoHeadDim = 128
        public var audioHeadDim = 64
        public var layers = 8
        public var registers = 128
        public var maxPosition = 4096
        public var ffMult = 4
    }

    public var transformer = Transformer()
    public var connector = Connector()

    /// Stage 1: eight steps of the distilled model at half resolution.
    public static let distilledSigmas: [Double] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    /// Stage 2: three steps at full resolution, from the upscaled stage-1 latent renoised at 0.909375.
    public static let stage2Sigmas: [Double] = [0.909375, 0.725, 0.421875, 0.0]
    /// The prompt is left-padded to this many tokens (`LTX2_GEMMA_MAX_LENGTH`).
    public static let maxPromptTokens = 1024
    /// The video VAE compresses 8 frames and 32 × 32 pixels into one latent token.
    public static let temporalScale = 8
    public static let spatialScale = 32
    /// Audio latents per second: 16 kHz, hop 160, downsampled 4 times.
    public static let audioLatentsPerSecond = 25.0
    public static let latentChannels = 128
    /// The noise of stage 2 and of the ancestral sampler (LTX-2.5's stage 1) are drawn from these
    /// seed offsets.
    public static let stage2SeedOffset = 2
    public static let ancestralSeedOffset = 10000

    /// LTX-2.5 (`is_ltx25_pack`: the video feed-forward has no bias): its own Gemma 4 text encoder,
    /// the ancestral sampler in stage 1, the keyframe marker, renamed VAE and upsampler files.
    public var isLTX25: Bool { !transformer.ffBias }

    public init() {}

    /// Reads the transformer's settings from the pack (`embedded_config.json`, else `config.json`),
    /// keeping the defaults above for what they do not say.
    public static func load(pack: URL) -> LTXConfig {
        var config = LTXConfig()
        for name in ["embedded_config.json", "config.json"] {
            guard let data = try? Data(contentsOf: pack.appending(path: name)),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            let t = (json["transformer"] as? [String: Any]) ?? json
            var c = config.transformer
            c.numLayers = t["num_layers"] as? Int ?? c.numLayers
            c.videoDim = t["cross_attention_dim"] as? Int ?? c.videoDim
            c.audioDim = t["audio_cross_attention_dim"] as? Int ?? c.audioDim
            c.videoHeads = t["num_attention_heads"] as? Int ?? c.videoHeads
            c.audioHeads = t["audio_num_attention_heads"] as? Int ?? c.audioHeads
            c.videoHeadDim = t["attention_head_dim"] as? Int ?? c.videoHeadDim
            c.audioHeadDim = t["audio_attention_head_dim"] as? Int ?? c.audioHeadDim
            c.crossHeads = t["audio_num_attention_heads"] as? Int ?? c.crossHeads
            c.crossHeadDim = t["audio_attention_head_dim"] as? Int ?? c.crossHeadDim
            c.videoPatchChannels = t["in_channels"] as? Int ?? c.videoPatchChannels
            c.audioPatchChannels = t["audio_in_channels"] as? Int ?? c.audioPatchChannels
            if let value = t["timestep_scale_multiplier"] as? Double { c.timestepScale = Float(value) }
            if let value = t["av_ca_timestep_scale_multiplier"] as? Double { c.crossTimestepScale = Float(value) }
            if let value = t["positional_embedding_theta"] as? Double { c.ropeTheta = value }
            c.maxPositions = t["positional_embedding_max_pos"] as? [Int] ?? c.maxPositions
            c.audioMaxPositions = t["audio_positional_embedding_max_pos"] as? [Int] ?? c.audioMaxPositions
            if let value = t["norm_eps"] as? Double { c.normEps = Float(value) }
            c.ffBias = t["ff_bias"] as? Bool ?? c.ffBias
            c.audioFFBias = t["audio_ff_bias"] as? Bool ?? c.audioFFBias
            c.doublePrecisionRope = (t["frequencies_precision"] as? String) == "float64"
            c.keyframesEmbedding = t["use_keyframes_abs_pos_embedding"] as? Bool ?? false
            config.transformer = c
            return config
        }
        return config
    }
}

/// The geometry of one clip: pixels, frames and the latent grid they map to.
public struct LTXGeometry: Equatable {
    public let width: Int
    public let height: Int
    public let frames: Int
    public let fps: Double

    /// Sides floored to multiples of 64 (stage 1 runs at half resolution, on a 32-pixel grid) and
    /// the frame count to 8k + 1, as the two-stage pipeline needs them.
    public init(width: Int, height: Int, frames: Int, fps: Double) {
        self.width = max(64, (width / 64) * 64)
        self.height = max(64, (height / 64) * 64)
        self.frames = max(1, ((frames - 1) / LTXConfig.temporalScale) * LTXConfig.temporalScale + 1)
        self.fps = fps
    }

    /// Latent frames: the first latent frame covers one pixel frame, every later one eight.
    public var latentFrames: Int { (frames + LTXConfig.temporalScale - 1) / LTXConfig.temporalScale }
    public var latentHeight: Int { height / LTXConfig.spatialScale }
    public var latentWidth: Int { width / LTXConfig.spatialScale }
    /// `compute_audio_token_count`: 25 audio latents per second of video.
    public var audioTokens: Int { Int((Double(frames) / fps * LTXConfig.audioLatentsPerSecond).rounded(.toNearestOrEven)) }
}
