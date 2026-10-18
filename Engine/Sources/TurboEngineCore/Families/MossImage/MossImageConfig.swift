import Foundation

/// The geometry of a MOSS-Image checkpoint, read from its diffusers folders (`transformer/config.json`,
/// `text_encoder/config.json`, `vae/config.json`), so a fixture (see Fixtures/) shrinks it the same way.
/// The SFT and the Base share it; only the transformer's weights differ.
public struct MossImageConfig: Equatable, Sendable {
    /// `MOSSImageTransformer`'s arguments.
    public struct Transformer: Equatable, Sendable {
        public var dim = 3072
        public var ffnDim = 8192
        public var numHeads = 24
        public var numLayers = 42
        public var textDim = 4096
        public var freqDim = 256
        public var latentChannels = 128
        public var ropeBase: Float = 10000

        public var headDim: Int { dim / numHeads }
    }

    /// The `text_config` of Qwen3.5-9B (`Qwen3_5Model`), whose last hidden state is the prompt.
    public struct TextEncoder: Equatable, Sendable {
        public var vocabSize = 248_320
        public var hiddenSize = 4096
        public var intermediateSize = 12288
        public var numHiddenLayers = 32
        public var numAttentionHeads = 16
        public var numKeyValueHeads = 4
        public var headDim = 256
        public var rmsNormEps: Float = 1e-6
        public var ropeTheta: Float = 10_000_000
        public var partialRotaryFactor: Float = 0.25
        public var convKernel = 4
        public var linearKeyHeadDim = 128
        public var linearValueHeadDim = 128
        public var linearNumKeyHeads = 16
        public var linearNumValueHeads = 32
        /// Per layer: full attention or the gated delta net (every fourth is full attention).
        public var fullAttention: [Bool] = (0 ..< 32).map { ($0 + 1) % 4 == 0 }

        public var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) }
    }

    public var transformer = Transformer()
    public var textEncoder = TextEncoder()
    /// The decoder's widths: BFL's small decoder, narrower than FLUX.2's own encoder side.
    public var decoderChannels = [96, 192, 384, 384]
    /// `encode_prompt`'s truncation.
    public var maxSequenceLength = 4096
    public var defaultSteps = 50
    public var defaultGuidance = 3.0
    /// `MOSSImageScheduler`: the inference grid's shift and the training grid's.
    public var shift: Float = 5
    public var trainingShift: Float = 1
    public var numTrainTimesteps = 1000
    /// Blocks the negative branch of guidance skips (`uncond_skip_layers`, zero-based).
    public var unconditionalSkipLayers: Set<Int> = [22]

    public init() {}

    /// The configuration a checkpoint's folders describe; the defaults where a file or key is missing.
    public init(modelPath: URL) {
        func json(_ component: String) -> [String: Any] {
            let url = modelPath.appending(path: component).appending(path: "config.json")
            guard let data = try? Data(contentsOf: url) else { return [:] }
            return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        }
        func int(_ dict: [String: Any], _ key: String, _ fallback: Int) -> Int { (dict[key] as? NSNumber)?.intValue ?? fallback }
        func float(_ dict: [String: Any], _ key: String, _ fallback: Float) -> Float { (dict[key] as? NSNumber)?.floatValue ?? fallback }

        let t = json("transformer")
        transformer.dim = int(t, "dim", transformer.dim)
        transformer.ffnDim = int(t, "ffn_dim", transformer.ffnDim)
        transformer.numHeads = int(t, "num_heads", transformer.numHeads)
        transformer.numLayers = int(t, "num_layers", transformer.numLayers)
        transformer.textDim = int(t, "text_dim", transformer.textDim)
        transformer.freqDim = int(t, "freq_dim", transformer.freqDim)
        transformer.latentChannels = int(t, "latent_channels", transformer.latentChannels)
        transformer.ropeBase = float(t, "rope_base", transformer.ropeBase)

        let root = json("text_encoder")
        let e = root["text_config"] as? [String: Any] ?? root
        textEncoder.vocabSize = int(e, "vocab_size", textEncoder.vocabSize)
        textEncoder.hiddenSize = int(e, "hidden_size", textEncoder.hiddenSize)
        textEncoder.intermediateSize = int(e, "intermediate_size", textEncoder.intermediateSize)
        textEncoder.numHiddenLayers = int(e, "num_hidden_layers", textEncoder.numHiddenLayers)
        textEncoder.numAttentionHeads = int(e, "num_attention_heads", textEncoder.numAttentionHeads)
        textEncoder.numKeyValueHeads = int(e, "num_key_value_heads", textEncoder.numKeyValueHeads)
        textEncoder.headDim = int(e, "head_dim", textEncoder.headDim)
        textEncoder.rmsNormEps = float(e, "rms_norm_eps", textEncoder.rmsNormEps)
        textEncoder.convKernel = int(e, "linear_conv_kernel_dim", textEncoder.convKernel)
        textEncoder.linearKeyHeadDim = int(e, "linear_key_head_dim", textEncoder.linearKeyHeadDim)
        textEncoder.linearValueHeadDim = int(e, "linear_value_head_dim", textEncoder.linearValueHeadDim)
        textEncoder.linearNumKeyHeads = int(e, "linear_num_key_heads", textEncoder.linearNumKeyHeads)
        textEncoder.linearNumValueHeads = int(e, "linear_num_value_heads", textEncoder.linearNumValueHeads)
        let rope = e["rope_parameters"] as? [String: Any] ?? e
        textEncoder.ropeTheta = float(rope, "rope_theta", textEncoder.ropeTheta)
        textEncoder.partialRotaryFactor = float(rope, "partial_rotary_factor", float(e, "partial_rotary_factor", textEncoder.partialRotaryFactor))
        if let types = e["layer_types"] as? [String] {
            textEncoder.fullAttention = types.map { $0 == "full_attention" }
        } else {
            let interval = int(e, "full_attention_interval", 4)
            textEncoder.fullAttention = (0 ..< textEncoder.numHiddenLayers).map { ($0 + 1) % interval == 0 }
        }

        let v = json("vae")
        if let channels = (v["decoder_block_out_channels"] ?? v["block_out_channels"]) as? [NSNumber] {
            decoderChannels = channels.map(\.intValue)
        }
        let s = json("scheduler")
        shift = float(s, "shift", shift)
        trainingShift = float(s, "training_shift", trainingShift)
        numTrainTimesteps = int(s, "num_train_timesteps", numTrainTimesteps)
    }
}

/// `MOSSImageScheduler.set_timesteps`: the inference grid (shifted by `shift`) snapped to the nearest
/// point of the training grid (shifted by `trainingShift`), repeats dropped, then diffusers' Euler
/// scheduler with no further shift: model timesteps σ · 1000, a terminal 0. All in float32, with
/// torch's CPU `linspace` reproduced so the snapped sigmas are the reference's to the bit.
public struct MossImageSchedule: Equatable, Sendable {
    /// One per step, then the terminal 0.
    public let sigmas: [Float]
    /// σ · 1000 for each step, as the model reads it.
    public let timesteps: [Float]

    public var steps: Int { timesteps.count }

    public init(steps: Int, config: MossImageConfig = MossImageConfig()) {
        let total = config.numTrainTimesteps
        let inference = Self.grid(steps, shift: config.shift, total: total)
        let training = Array(Self.grid(total, shift: config.trainingShift, total: total).reversed())
        var snapped: [Float] = []
        for sigma in inference {
            // torch.searchsorted (left side), clamped to the last index.
            var low = 0, high = training.count
            while low < high {
                let middle = (low + high) / 2
                if training[middle] < sigma { low = middle + 1 } else { high = middle }
            }
            let upper = min(low, total - 1)
            let lower = max(upper - 1, 0)
            let nearest = abs(sigma - training[lower]) <= abs(training[upper] - sigma) ? lower : upper
            if snapped.last != training[nearest] { snapped.append(training[nearest]) }
        }
        sigmas = snapped + [0]
        timesteps = snapped.map { $0 * Float(total) }
    }

    /// `grid(count, shift)`: `shift · v / (1 + (shift − 1) · v)` over `linspace(1, lower, count)`.
    static func grid(_ count: Int, shift: Float, total: Int) -> [Float] {
        // `lower` is a Python float; linspace rounds both ends to float32.
        let lower = Float(Double(shift) / (Double(total) + Double(shift) - 1))
        return linspace(1, lower, count).map { shift * $0 / (1 + (shift - 1) * $0) }
    }

    /// torch.linspace on the CPU in float32: the step in float32, the first half counted up from the
    /// start and the second down from the end, each a fused multiply-add.
    static func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        let halfway = count / 2
        return (0 ..< count).map { i in
            i < halfway ? start.addingProduct(step, Float(i)) : end.addingProduct(-step, Float(count - i - 1))
        }
    }
}
