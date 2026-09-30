import Foundation
import MLX
import MLXNN

// LTX-2.5's text encoder: the Gemma 4 12B text tower its packs carry (`text_encoder.safetensors`,
// keys under `text_encoder.model.`), as dgrauet's `Gemma4TextModel` runs it (`gemma4.py`): the
// embedding and the output of each of the 48 layers, the last one through the final norm. What
// sets it apart from Gemma 3: norms multiply by the weight itself (no `1 + weight`) in float32,
// attention has no scale (q_norm and k_norm carry it) and a scale-free v_norm, the full-attention
// layers have 512-wide heads, one KV head, no v_proj (the values are the raw k_proj output) and a
// "proportional" RoPE over a quarter of the head, each layer ends with a learned scalar, and the
// MLP's GELU is the exact tanh formula.

struct Gemma4Config {
    var hiddenSize = 3840
    var intermediateSize = 15360
    var heads = 16
    var headDim = 256
    var globalHeadDim = 512
    var kvHeads = 8
    var globalKVHeads = 1
    var layers = 48
    var rmsNormEps: Float = 1e-6
    var slidingWindow = 1024
    var layerTypes: [String] = (0 ..< 48).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" }
    var kEqualsV = true
    var localTheta: Float = 10_000
    var globalTheta: Float = 1_000_000
    var partialRotaryFactor: Float = 0.25
    var vocabSize = 262_144
    var padTokenId = 0

    func isSliding(_ layer: Int) -> Bool { layerTypes[layer] == "sliding_attention" }

    /// `Gemma4TextConfig.from_text_encoder_config`: the `text_config` of the pack's
    /// `text_encoder_config.json`.
    static func load(pack: URL) throws -> Gemma4Config {
        var config = Gemma4Config()
        let data = try Data(contentsOf: pack.appending(path: "text_encoder_config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = json["text_config"] as? [String: Any] else { return config }
        config.hiddenSize = text["hidden_size"] as? Int ?? config.hiddenSize
        config.intermediateSize = text["intermediate_size"] as? Int ?? config.intermediateSize
        config.heads = text["num_attention_heads"] as? Int ?? config.heads
        config.headDim = text["head_dim"] as? Int ?? config.headDim
        config.globalHeadDim = text["global_head_dim"] as? Int ?? config.globalHeadDim
        config.kvHeads = text["num_key_value_heads"] as? Int ?? config.kvHeads
        config.globalKVHeads = text["num_global_key_value_heads"] as? Int ?? config.globalKVHeads
        config.layers = text["num_hidden_layers"] as? Int ?? config.layers
        if let value = text["rms_norm_eps"] as? Double { config.rmsNormEps = Float(value) }
        config.slidingWindow = text["sliding_window"] as? Int ?? config.slidingWindow
        config.layerTypes = text["layer_types"] as? [String] ?? config.layerTypes
        config.kEqualsV = text["attention_k_eq_v"] as? Bool ?? false
        config.vocabSize = text["vocab_size"] as? Int ?? config.vocabSize
        config.padTokenId = text["pad_token_id"] as? Int ?? config.padTokenId
        if let rope = text["rope_parameters"] as? [String: Any] {
            if let sliding = rope["sliding_attention"] as? [String: Any], let theta = sliding["rope_theta"] as? Double {
                config.localTheta = Float(theta)
            }
            if let full = rope["full_attention"] as? [String: Any] {
                if let theta = full["rope_theta"] as? Double { config.globalTheta = Float(theta) }
                if let factor = full["partial_rotary_factor"] as? Double { config.partialRotaryFactor = Float(factor) }
            }
        }
        return config
    }
}

/// `Gemma4RMSNorm`: normalized in float32, times the weight (not `1 + weight`), cast back.
final class Gemma4RMSNorm: Module {
    @ParameterInfo var weight: MLXArray?
    let eps: Float

    init(dimensions: Int, eps: Float, withScale: Bool = true) {
        self.eps = eps
        _weight.wrappedValue = withScale ? MLXArray.ones([dimensions]) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        var normed = x32 * rsqrt(mean(square(x32), axis: -1, keepDims: true) + eps)
        if let weight { normed = normed * weight.asType(.float32) }
        return normed.asType(x.dtype)
    }
}

/// `apply_rotary_pos_emb` on [B, T, H, D] with float32 tables [T, D] cast to the input's dtype.
private func gemma4Rotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let half = x.shape[3] / 2
    let c = cos.reshaped([1, cos.shape[0], 1, cos.shape[1]]).asType(x.dtype)
    let s = sin.reshaped([1, sin.shape[0], 1, sin.shape[1]]).asType(x.dtype)
    let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    return x * c + rotated * s
}

final class Gemma4Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNorm

    let heads: Int
    let kvHeads: Int
    let headDim: Int

    init(config: Gemma4Config, layer: Int) {
        let sliding = config.isSliding(layer)
        heads = config.heads
        kvHeads = sliding ? config.kvHeads : config.globalKVHeads
        headDim = sliding ? config.headDim : config.globalHeadDim
        let kEqualsV = config.kEqualsV && !sliding
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = kEqualsV ? nil : Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = Gemma4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Gemma4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = Gemma4RMSNorm(dimensions: headDim, eps: config.rmsNormEps, withScale: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        var q = qProj(x).reshaped([batch, length, heads, headDim])
        q = gemma4Rotary(qNorm(q), cos: cos, sin: sin).transposed(0, 2, 1, 3)
        let rawK = kProj(x).reshaped([batch, length, kvHeads, headDim])
        // On the k = v layers the values are the raw k_proj output, before k_norm and RoPE.
        let rawV = vProj.map { $0(x).reshaped([batch, length, kvHeads, headDim]) } ?? rawK
        var k = gemma4Rotary(kNorm(rawK), cos: cos, sin: sin).transposed(0, 2, 1, 3)
        var v = vNorm(rawV).transposed(0, 2, 1, 3)
        // `repeat_kv`: each KV head repeated contiguously.
        let groups = heads / kvHeads
        if groups > 1 {
            k = broadcast(k.expandedDimensions(axis: 2), to: [batch, kvHeads, groups, length, headDim])
                .reshaped([batch, heads, length, headDim])
            v = broadcast(v.expandedDimensions(axis: 2), to: [batch, kvHeads, groups, length, headDim])
                .reshaped([batch, heads, length, headDim])
        }
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1, mask: mask)
        return oProj(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }
}

final class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Gemma4Config) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    /// `gelu_pytorch_tanh`, written out as the reference writes it.
    static func geluTanh(_ x: MLXArray) -> MLXArray {
        0.5 * x * (1 + tanh(Float((2.0 / Double.pi).squareRoot()) * (x + 0.044715 * pow(x, 3))))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(Self.geluTanh(gateProj(x)) * upProj(x))
    }
}

final class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma4RMSNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4Config, layer: Int) {
        _selfAttn.wrappedValue = Gemma4Attention(config: config, layer: layer)
        _mlp.wrappedValue = Gemma4MLP(config: config)
        _inputLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + postAttentionLayerNorm(selfAttn(inputLayerNorm(x), cos: cos, sin: sin, mask: mask))
        let out = h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        return out * layerScalar
    }
}

/// `Gemma4TextModel` (`text_encoder.model` in the pack).
final class Gemma4TextModel: Module, LTXTextTower {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNorm

    let config: Gemma4Config

    init(config: Gemma4Config) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layers).map { Gemma4DecoderLayer(config: config, layer: $0) }
        _norm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// `Gemma4RotaryEmbedding`: float32 cos and sin [T, D]. "proportional" fills only the first
    /// `factor · D / 2` frequencies (still over D), the rest are zero, identity-rotated.
    static func rotaryTables(length: Int, headDim: Int, theta: Float, rotaryFactor: Float?) -> (MLXArray, MLXArray) {
        let half = headDim / 2
        let angles = rotaryFactor.map { Int($0 * Float(headDim)) / 2 } ?? half
        let exponents = MLXArray(stride(from: 0, to: 2 * angles, by: 2).map { Float($0) }) / Float(headDim)
        var inverse = 1 / pow(MLXArray(theta), exponents)
        if angles < half { inverse = concatenated([inverse, MLXArray.zeros([half - angles], dtype: .float32)]) }
        let positions = MLXArray(0 ..< length).asType(.float32)
        let freqs = positions.expandedDimensions(axis: 1) * inverse.expandedDimensions(axis: 0)
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb), sin(emb))
    }

    /// `build_attention_mask`: 0 where a key is visible (causal, not padding, within the window on
    /// sliding layers), −inf elsewhere, a fully masked row's diagonal unmasked.
    static func attentionMask(length: Int, window: Int?, padding: MLXArray, dtype: DType) -> MLXArray {
        let q = MLXArray(0 ..< length).reshaped([length, 1])
        let k = MLXArray(0 ..< length).reshaped([1, length])
        var visible = k .<= q
        if let window { visible = visible .&& (k .> (q - window)) }
        visible = visible.reshaped([1, 1, length, length]) .&& (padding .!= 0).reshaped([padding.shape[0], 1, 1, length])
        let rowMasked = .!(visible.any(axis: -1, keepDims: true))
        let diagonal = (k .== q).reshaped([1, 1, length, length])
        visible = visible .|| (rowMasked .&& diagonal)
        return MLX.where(visible, MLXArray(Float(0)).asType(dtype), MLXArray(-Float.infinity).asType(dtype))
    }

    /// The embedding (scaled by √hidden in its dtype) and the output of every layer, the last
    /// through the final norm: 49 arrays [B, T, hidden]. Each layer is evaluated as it finishes.
    func allHiddenStates(tokens: MLXArray, attentionMask: MLXArray) -> [MLXArray] {
        var h = embedTokens(tokens)
        h = h * MLXArray(Float(config.hiddenSize).squareRoot()).asType(h.dtype)
        var states = [h]

        let length = tokens.shape[1]
        let local = Self.rotaryTables(length: length, headDim: config.headDim, theta: config.localTheta, rotaryFactor: nil)
        let global = Self.rotaryTables(length: length, headDim: config.globalHeadDim, theta: config.globalTheta,
                                       rotaryFactor: config.partialRotaryFactor)
        let fullMask = Self.attentionMask(length: length, window: nil, padding: attentionMask, dtype: h.dtype)
        let slidingMask = length <= config.slidingWindow ? fullMask
            : Self.attentionMask(length: length, window: config.slidingWindow, padding: attentionMask, dtype: h.dtype)

        for (index, layer) in layers.enumerated() {
            let sliding = config.isSliding(index)
            let (c, s) = sliding ? local : global
            h = layer(h, cos: c, sin: s, mask: sliding ? slidingMask : fullMask)
            states.append(h)
            eval(h)
        }
        states[states.count - 1] = norm(h)
        return states
    }

    static let prefix = "text_encoder.model."

    /// The tower's keys without their prefix.
    static func weights(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for (key, value) in tensors where key.hasPrefix(prefix) {
            weights[String(key.dropFirst(prefix.count))] = value
        }
        return weights
    }
}
