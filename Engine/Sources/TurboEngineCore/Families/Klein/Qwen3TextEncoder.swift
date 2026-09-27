import Foundation
import MLX
import MLXNN

// FLUX.2 Klein's text encoder: a Qwen3 decoder stack read for its hidden states. Module and
// parameter names follow mflux (`flux2_text_encoder/qwen3_text_encoder.py` and
// `common_models/qwen3_vl/*`), so the checkpoint's keys load unchanged.

/// RMSNorm computed in float32 with the weight applied in float32, as `Qwen3VLRMSNorm` does.
final class Qwen3RMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputType = x.dtype
        let x32 = x.asType(.float32)
        let variance = square(x32).mean(axis: -1, keepDims: true)
        let normalized = x32 * rsqrt(variance + eps)
        return (weight.asType(.float32) * normalized).asType(inputType)
    }
}

final class Qwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class Qwen3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen3RMSNorm

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    init(config: KleinConfig.TextEncoder) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = 1 / Float(config.headDim).squareRoot()
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = Qwen3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Qwen3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// `mask` is additive, [B, 1, S, S]; `cos`/`sin` are [1, S, headDim].
    func callAsFunction(_ x: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        var q = qProj(x).reshaped([batch, length, heads, headDim])
        var k = kProj(x).reshaped([batch, length, kvHeads, headDim])
        var v = vProj(x).reshaped([batch, length, kvHeads, headDim])
        q = qNorm(q).transposed(0, 2, 1, 3)
        k = kNorm(k).transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // HF-style rotary: cos and sin over the full head, halves rotated.
        let cosB = cos.expandedDimensions(axis: 1)
        let sinB = sin.expandedDimensions(axis: 1)
        q = q * cosB + Self.rotateHalf(q) * sinB
        k = k * cosB + Self.rotateHalf(k) * sinB

        if kvHeads != heads {
            let repeats = heads / kvHeads
            k = repeated(k, count: repeats, axis: 1)
            v = repeated(v, count: repeats, axis: 1)
        }

        // The reference attends in float32 and casts the result back.
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32), keys: k.asType(.float32), values: v.asType(.float32),
            scale: scale, mask: mask.asType(.float32)
        ).asType(q.dtype)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim])
        return oProj(merged)
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.shape[x.ndim - 1] / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half...]
        return concatenated([-x2, x1], axis: -1)
    }
}

final class Qwen3DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Qwen3RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Qwen3RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP

    init(config: KleinConfig.TextEncoder) {
        _inputLayerNorm.wrappedValue = Qwen3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfAttn.wrappedValue = Qwen3Attention(config: config)
        _postAttentionLayerNorm.wrappedValue = Qwen3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = Qwen3MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask, cos: cos, sin: sin)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}

/// The encoder: token embeddings, the decoder layers, and the hidden states after each of them.
public final class Qwen3TextEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Qwen3RMSNorm

    let config: KleinConfig.TextEncoder

    public init(config: KleinConfig.TextEncoder) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in Qwen3DecoderLayer(config: config) }
        _norm.wrappedValue = Qwen3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Keys of the checkpoint this module has no parameter for (the rotary table is computed).
    public static func ignoresKey(_ key: String) -> Bool {
        key.hasPrefix("rotary_emb.")
    }

    /// The hidden states, embeddings first, then one per layer (before the final norm), as the
    /// reference returns them with `output_hidden_states`.
    public func hiddenStates(inputIds: MLXArray, attentionMask: MLXArray) -> [MLXArray] {
        let (batch, length) = (inputIds.shape[0], inputIds.shape[1])
        var h = embedTokens(inputIds)
        let dtype = h.dtype

        // Padding (keys of padded tokens) plus causality, additive.
        let padding = MLX.where(attentionMask .== 1, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(dtype)
            .reshaped([batch, 1, 1, length])
        let positions = MLXArray(0 ..< Int32(length))
        let causalBool = positions.expandedDimensions(axis: 0) .> positions.expandedDimensions(axis: 1)
        let causal = MLX.where(causalBool, MLXArray(-Float.infinity), MLXArray(Float(0)))
            .asType(dtype)
            .reshaped([1, 1, length, length])
        let mask = broadcast(causal, to: [batch, 1, length, length]) + padding

        let (cos, sin) = rotary(length: length, dtype: dtype)
        var states = [h]
        for layer in layers {
            h = layer(h, mask: mask, cos: cos, sin: sin)
            states.append(h)
        }
        return states
    }

    /// The concatenated hidden states of `layers` for every token: the prompt the transformer reads.
    public func promptEmbeds(inputIds: MLXArray, attentionMask: MLXArray, layers outLayers: [Int]) -> MLXArray {
        let states = hiddenStates(inputIds: inputIds, attentionMask: attentionMask)
        return concatenated(outLayers.map { states[$0] }, axis: -1)
    }

    /// `Qwen3TextRotaryEmbedding`: positions 0..<S, the frequencies duplicated over both halves.
    private func rotary(length: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let dim = config.headDim
        let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let invFreq = 1 / pow(Float(config.ropeTheta), exponents)
        let positions = MLXArray((0 ..< length).map { Float($0) }).reshaped([1, length, 1])
        let freqs = positions * invFreq.reshaped([1, 1, dim / 2])
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }
}
