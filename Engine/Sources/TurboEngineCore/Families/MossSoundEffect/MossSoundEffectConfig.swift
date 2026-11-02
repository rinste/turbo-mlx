import Foundation

/// MOSS-SoundEffect v2.0's settings, as its diffusers-style folders give them (`model_index.json`,
/// `transformer/config.json`, `scheduler/scheduler_config.json`) and as `MossSoundEffectPipeline` and
/// `WanAudioPipeline` in OpenMOSS's MOSS-TTS (`moss_soundeffect_v2/`) fix the rest. The DAC VAE's shape is the
/// shipped checkpoint's (`vae_128d_48k.pth`, a pickled model object that names no config): 48 kHz mono, 960
/// samples a latent (50 a second).
public struct MossSoundEffectConfig: Sendable {
    // The Wan-style audio DiT (`WanAudioModel`, the "1.3B" preset).
    public var dim = 1536
    public var ffnDim = 8960
    public var heads = 12
    public var layers = 30
    public var latentChannels = 128
    public var textDim = 2048
    public var freqDim = 256
    public var eps: Float = 1e-6

    // The DAC VAE's decoder (`decoder_dim`, `decoder_rates`).
    public var decoderDim = 2048
    public var decoderRates = [8, 5, 4, 3, 2]
    public var sampleRate = 48_000
    public var hopLength: Int { decoderRates.reduce(1, *) }

    /// Every clip is denoised as this many seconds of latents and cut to the length asked for.
    public var maxSeconds = 30
    /// `WanPrompter(text_len=512)`: the prompt padded (or cut) to this many tokens.
    public var textLength = 512
    /// `FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)`, 1000 training timesteps.
    public var shift = 5.0
    public var trainTimesteps = 1000

    public static let defaultSteps = 100
    public static let defaultGuidance = 4.0
    public static let defaultSeconds = 10.0
    public static let maxSteps = 200

    /// The latent frames of the whole denoised span (1500 for 30 s).
    public var latentFrames: Int { sampleRate * maxSeconds / hopLength }

    /// Qwen3-1.7B, the frozen text encoder.
    public var textEncoder: KleinConfig.TextEncoder = {
        var encoder = KleinConfig.TextEncoder(hiddenSize: 2048, intermediateSize: 6144)
        encoder.numHiddenLayers = 28
        encoder.numAttentionHeads = 16
        encoder.numKeyValueHeads = 8
        return encoder
    }()

    public init() {}

    /// The folder's own numbers where it has them (a fixture's smaller ones too).
    public static func read(folder: URL) -> MossSoundEffectConfig {
        var config = MossSoundEffectConfig()
        func json(_ path: String) -> [String: Any] {
            guard let data = try? Data(contentsOf: folder.appending(path: path)),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
            return object
        }
        let index = json("model_index.json")
        if let rate = index["sample_rate"] as? Int { config.sampleRate = rate }
        if let seconds = index["max_inference_seconds"] as? Int { config.maxSeconds = seconds }
        let transformer = json("transformer/config.json")
        if let value = transformer["dim"] as? Int { config.dim = value }
        if let value = transformer["ffn_dim"] as? Int { config.ffnDim = value }
        if let value = transformer["num_heads"] as? Int { config.heads = value }
        if let value = transformer["num_layers"] as? Int { config.layers = value }
        if let value = transformer["in_dim"] as? Int { config.latentChannels = value }
        if let value = transformer["text_dim"] as? Int { config.textDim = value }
        if let value = transformer["freq_dim"] as? Int { config.freqDim = value }
        if let value = transformer["eps"] as? Double { config.eps = Float(value) }
        let scheduler = json("scheduler/scheduler_config.json")
        if let value = scheduler["shift"] as? Double { config.shift = value }
        if let value = scheduler["num_train_timesteps"] as? Int { config.trainTimesteps = value }
        let encoder = json("text_encoder/config.json")
        if let value = encoder["hidden_size"] as? Int { config.textEncoder.hiddenSize = value }
        if let value = encoder["intermediate_size"] as? Int { config.textEncoder.intermediateSize = value }
        if let value = encoder["num_hidden_layers"] as? Int { config.textEncoder.numHiddenLayers = value }
        if let value = encoder["num_attention_heads"] as? Int { config.textEncoder.numAttentionHeads = value }
        if let value = encoder["num_key_value_heads"] as? Int { config.textEncoder.numKeyValueHeads = value }
        if let value = encoder["head_dim"] as? Int { config.textEncoder.headDim = value }
        if let value = encoder["vocab_size"] as? Int { config.textEncoder.vocabSize = value }
        if let value = encoder["rope_theta"] as? Double { config.textEncoder.ropeTheta = value }
        if let value = encoder["rms_norm_eps"] as? Double {
            config.textEncoder.rmsNormEps = Float(value)
            config.textEncoder.qkNormEps = Float(value)
        }
        return config
    }
}
