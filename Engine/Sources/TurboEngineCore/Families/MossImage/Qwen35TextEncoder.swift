import Foundation
import MLX
import MLXNN

// MOSS-Image's text encoder: the language model of Qwen3.5-9B (transformers' `Qwen3_5TextModel`),
// run once over the whole prompt for its last hidden state. Three layers in four are a gated delta
// net (linear attention with a recurrent state), the fourth gated full attention. Module and
// parameter names follow the checkpoint's `language_model.*` keys; the vision tower is not read.

/// `Qwen3_5RMSNorm`: zero-centred, `x̂ · (1 + w)` in float32, then cast back.
final class Qwen35RMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: 1 + weight.asType(.float32), eps: eps).asType(x.dtype)
    }
}

/// `Qwen3_5RMSNormGated`: `w · x̂` in the input's dtype, times `silu(gate)` in float32.
final class Qwen35RMSNormGated: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let normed = MLXFast.rmsNorm(x.asType(.float32), weight: MLXArray.ones([x.dim(-1)]), eps: eps).asType(dtype)
        return ((weight.asType(dtype) * normed).asType(.float32) * silu(gate.asType(.float32))).asType(dtype)
    }
}

/// The depthwise causal convolution in front of the delta net, its weight in torch's [C, 1, K].
final class Qwen35CausalConv: Module {
    @ParameterInfo var weight: MLXArray

    init(channels: Int, kernel: Int) {
        _weight.wrappedValue = MLXArray.zeros([channels, 1, kernel])
        super.init()
    }

    /// [B, S, C] → [B, S, C]: each channel's last K inputs (zeros before the first), summed in
    /// float32, cast back, then `silu` as `causal_conv1d_fn` applies it.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (channels, kernel) = (weight.dim(0), weight.dim(2))
        let length = x.dim(1)
        let w = weight.reshaped([channels, kernel]).asType(.float32)
        let padded = MLX.padded(x.asType(.float32), widths: [IntOrPair(0), IntOrPair((kernel - 1, 0)), IntOrPair(0)])
        var sum = padded[0..., 0 ..< length, 0...] * w[0..., 0]
        for k in 1 ..< kernel {
            sum = sum + padded[0..., k ..< (k + length), 0...] * w[0..., k]
        }
        return silu(sum.asType(x.dtype))
    }
}

/// The gated delta rule's recurrence in one Metal kernel: per (batch, value head) a [Dv, Dk] state,
/// each thread of a simdgroup holding Dk/32 entries of one state row. Inputs are float32 (q and k
/// already l2-normalized, q scaled), `decay` = exp(g); the key head of value head `hv` is
/// `hv / (Hv / Hk)`, as `repeat_interleave` pairs them. Laid out as mlx-lm's `gated_delta` kernel;
/// the chunked form transformers runs is the same sum in another order.
enum GatedDeltaRule {
    static let kernel = MLXFast.metalKernel(
        name: "turbo_gated_delta_rule",
        inputNames: ["q", "k", "v", "decay", "beta", "T"],
        outputNames: ["y"],
        source: """
            auto n = thread_position_in_grid.z;
            auto b = n / Hv;
            auto hv = n % Hv;
            auto hk = hv / (Hv / Hk);
            constexpr int per = Dk / 32;
            auto lane = thread_position_in_threadgroup.x;
            auto dv = thread_position_in_grid.y;

            auto q_ = q + b * T * Hk * Dk + hk * Dk;
            auto k_ = k + b * T * Hk * Dk + hk * Dk;
            auto v_ = v + b * T * Hv * Dv + hv * Dv;
            auto y_ = y + b * T * Hv * Dv + hv * Dv;
            auto decay_ = decay + b * T * Hv + hv;
            auto beta_ = beta + b * T * Hv + hv;

            float state[per];
            for (int i = 0; i < per; ++i) { state[i] = 0.0f; }
            for (int t = 0; t < T; ++t) {
                float g = decay_[t * Hv];
                float memory = 0.0f;
                for (int i = 0; i < per; ++i) {
                    state[i] = state[i] * g;
                    memory += state[i] * k_[t * Hk * Dk + per * lane + i];
                }
                memory = simd_sum(memory);
                float delta = (v_[t * Hv * Dv + dv] - memory) * beta_[t * Hv];
                float out = 0.0f;
                for (int i = 0; i < per; ++i) {
                    state[i] = state[i] + k_[t * Hk * Dk + per * lane + i] * delta;
                    out += state[i] * q_[t * Hk * Dk + per * lane + i];
                }
                out = simd_sum(out);
                if (lane == 0) { y_[t * Hv * Dv + dv] = out; }
            }
            """
    )

    /// q, k [B, T, Hk, Dk], v [B, T, Hv, Dv], decay and beta [B, T, Hv], all float32 → [B, T, Hv, Dv].
    static func callAsFunction(q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray) -> MLXArray {
        let (batch, length, keyHeads, keyDim) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        let (valueHeads, valueDim) = (v.dim(2), v.dim(3))
        precondition(keyDim % 32 == 0 && valueHeads % keyHeads == 0)
        return kernel(
            [q, k, v, decay, beta, MLXArray(Int32(length))],
            template: [("Dk", keyDim), ("Dv", valueDim), ("Hk", keyHeads), ("Hv", valueHeads)],
            grid: (32, valueDim, batch * valueHeads),
            threadGroup: (32, 4, 1),
            outputShapes: [[batch, length, valueHeads, valueDim]],
            outputDTypes: [.float32]
        )[0]
    }
}

/// `Qwen3_5GatedDeltaNet` without a cache: the whole prompt at once.
final class Qwen35GatedDeltaNet: Module {
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Qwen35CausalConv
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen35RMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let keyHeads, valueHeads, keyDim, valueDim: Int

    init(config: MossImageConfig.TextEncoder) {
        keyHeads = config.linearNumKeyHeads
        valueHeads = config.linearNumValueHeads
        keyDim = config.linearKeyHeadDim
        valueDim = config.linearValueHeadDim
        let convDim = 2 * keyHeads * keyDim + valueHeads * valueDim
        _inProjQKV.wrappedValue = Linear(config.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(config.hiddenSize, valueHeads * valueDim, bias: false)
        _inProjB.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _inProjA.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _conv1d.wrappedValue = Qwen35CausalConv(channels: convDim, kernel: config.convKernel)
        _dtBias.wrappedValue = MLXArray.ones([valueHeads])
        _aLog.wrappedValue = MLXArray.zeros([valueHeads])
        _norm.wrappedValue = Qwen35RMSNormGated(dimensions: valueDim, eps: config.rmsNormEps)
        _outProj.wrappedValue = Linear(valueHeads * valueDim, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let dtype = x.dtype
        let mixed = conv1d(inProjQKV(x))
        let keyWidth = keyHeads * keyDim
        let q = mixed[0..., 0..., 0 ..< keyWidth].reshaped([batch, length, keyHeads, keyDim])
        let k = mixed[0..., 0..., keyWidth ..< (2 * keyWidth)].reshaped([batch, length, keyHeads, keyDim])
        let v = mixed[0..., 0..., (2 * keyWidth)...].reshaped([batch, length, valueHeads, valueDim])
        let z = inProjZ(x).reshaped([batch, length, valueHeads, valueDim])

        let beta = sigmoid(inProjB(x)).asType(.float32)
        let g = -exp(aLog.asType(.float32)) * softplus(inProjA(x).asType(.float32) + dtBias.asType(.float32))
        // `l2norm` in float32 (the eps on the sum of squares, as FLA's), q scaled by Dk^-½.
        func l2norm(_ x: MLXArray) -> MLXArray { x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6) }
        let query = l2norm(q.asType(.float32)) * (1 / Float(keyDim).squareRoot())
        let key = l2norm(k.asType(.float32))
        let out = GatedDeltaRule.callAsFunction(q: query, k: key, v: v.asType(.float32), decay: exp(g), beta: beta)
        let normed = norm(out.asType(dtype), gate: z)
        return outProj(normed.reshaped([batch, length, valueHeads * valueDim]))
    }
}

/// `Qwen3_5Attention`: q_proj gives each head's query and an output gate; zero-centred q/k norms;
/// rotary over the first quarter of the head; causal.
final class Qwen35Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen35RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen35RMSNorm

    let heads, kvHeads, headDim: Int

    init(config: MossImageConfig.TextEncoder) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = Qwen35RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Qwen35RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// `cos`/`sin` [1, S, rotaryDim] in the activations' dtype.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let projected = qProj(x).reshaped([batch, length, heads, 2 * headDim])
        let gate = projected[.ellipsis, headDim...].reshaped([batch, length, heads * headDim])
        var q = qNorm(projected[.ellipsis, 0 ..< headDim]).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped([batch, length, kvHeads, headDim])).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        q = Self.rotated(q, cos: cos, sin: sin)
        k = Self.rotated(k, cos: cos, sin: sin)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .causal)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim])
        return oProj(merged * sigmoid(gate))
    }

    /// `apply_rotary_pos_emb` with a partial rotary: the first `rotaryDim` features turn, the rest pass.
    static func rotated(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let rotary = cos.dim(-1)
        let turned = x[.ellipsis, 0 ..< rotary]
        let half = rotary / 2
        let rotatedHalf = concatenated([-turned[.ellipsis, half...], turned[.ellipsis, 0 ..< half]], axis: -1)
        let c = cos.expandedDimensions(axis: 1)
        let s = sin.expandedDimensions(axis: 1)
        return concatenated([turned * c + rotatedHalf * s, x[.ellipsis, rotary...]], axis: -1)
    }
}

final class Qwen35MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class Qwen35DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?
    @ModuleInfo(key: "mlp") var mlp: Qwen35MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Qwen35RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Qwen35RMSNorm

    init(config: MossImageConfig.TextEncoder, fullAttention: Bool) {
        if fullAttention {
            _selfAttn.wrappedValue = Qwen35Attention(config: config)
        } else {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(config: config)
        }
        _mlp.wrappedValue = Qwen35MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = Qwen35RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Qwen35RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let normed = inputLayerNorm(x)
        let mixed = selfAttn.map { $0(normed, cos: cos, sin: sin) } ?? linearAttn!(normed)
        let h = x + mixed
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The language model: embeddings, the hybrid layers, the final norm.
public final class Qwen35TextEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Qwen35RMSNorm

    let config: MossImageConfig.TextEncoder

    public init(config: MossImageConfig.TextEncoder) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = config.fullAttention.map { Qwen35DecoderLayer(config: config, fullAttention: $0) }
        _norm.wrappedValue = Qwen35RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// The checkpoint's language-model tensors under the module's names: `language_model.` (or
    /// `model.language_model.`) dropped, the vision tower and any multi-token-prediction head left out.
    public static func keys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var kept: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            for prefix in ["model.language_model.", "language_model."] where name.hasPrefix(prefix) {
                name.removeFirst(prefix.count)
            }
            if name == key || name.hasPrefix("mtp.") || name.contains("rotary_emb.") { continue }
            kept[name] = value
        }
        return kept
    }

    /// `last_hidden_state` for token ids [1, S]: [1, S, hidden]. `computeType` overrides the
    /// activations' dtype (the parity check runs in float32). `states` gets each layer's output.
    public func lastHiddenState(_ ids: MLXArray, computeType: DType? = nil, states: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        var h = embedTokens(ids)
        if let computeType { h = h.asType(computeType) }
        let (cos, sin) = rotary(length: ids.dim(1), dtype: h.dtype)
        for (index, layer) in layers.enumerated() {
            h = layer(h, cos: cos, sin: sin)
            states?(index, h)
            // One layer at a time: a long prompt's activations never pile up.
            eval(h)
        }
        return norm(h)
    }

    /// `Qwen3_5TextRotaryEmbedding` for a text-only prompt (its three position axes equal, so the
    /// interleaved sections fall back to one): positions 0..<S, frequencies over the rotary part,
    /// float32 then cast.
    func rotary(length: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let dim = config.rotaryDim
        let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let invFreq = 1 / pow(MLXArray(config.ropeTheta), exponents)
        let positions = MLXArray((0 ..< length).map { Float($0) }).reshaped([1, length, 1])
        let freqs = positions * invFreq.reshaped([1, 1, dim / 2])
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }
}
