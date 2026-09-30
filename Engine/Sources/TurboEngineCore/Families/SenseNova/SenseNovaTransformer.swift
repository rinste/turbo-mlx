import Foundation
import MLX
import MLXNN

// SenseNova-U1.5's backbone (NEO-unify, `modeling_qwen3.py` of SenseTime's code): one Qwen3 layer
// stack whose every weight comes twice, once for the tokens it reads (the prompt, pictures) and once,
// suffixed `_mot_gen`, for the image it generates. The engine keeps the two as separate stacks of
// the same shape (the understanding one reads the prompt once into a key–value cache; the
// generation one runs every step), so each can leave memory on its own.
//
// Each head is split in two: the first half is rotated by the token's position in the sequence and
// normalized on its own, the second half is normalized on its own and its halves are rotated by the
// token's row and column (zero for text).

/// `Qwen3RMSNorm` as transformers computes it: normalized in float32, cast back, then scaled.
final class SenseNovaRMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let normalized = x32 * rsqrt(square(x32).mean(axis: -1, keepDims: true) + eps)
        return weight * normalized.asType(x.dtype)
    }
}

/// Positions of a sequence's tokens: the text position, the row and the column, each [S].
public struct SenseNovaPositions {
    public var t: [Int32]
    public var h: [Int32]
    public var w: [Int32]

    public var count: Int { t.count }

    /// Text tokens at positions `start ..< start + count`, row and column zero.
    public static func text(count: Int, start: Int = 0) -> SenseNovaPositions {
        SenseNovaPositions(t: (0 ..< count).map { Int32(start + $0) }, h: Array(repeating: 0, count: count), w: Array(repeating: 0, count: count))
    }

    /// An image's tokens, row by row, all at text position `t`.
    public static func image(rows: Int, columns: Int, t: Int) -> SenseNovaPositions {
        let count = rows * columns
        return SenseNovaPositions(
            t: Array(repeating: Int32(t), count: count),
            h: (0 ..< count).map { Int32($0 / columns) },
            w: (0 ..< count).map { Int32($0 % columns) }
        )
    }
}

/// The rotary tables of one attention: cos and sin for the text half and for the row and column
/// quarters, [1, 1, S, d/2] each, in the activations' dtype.
struct SenseNovaRotary {
    let cosT, sinT, cosH, sinH, cosW, sinW: MLXArray

    /// `Qwen3RotaryEmbedding`: frequencies base^(−2i/d) over a part of d dimensions, computed in
    /// float32, duplicated over both halves of the part, cast to the activations' dtype.
    init(positions: SenseNovaPositions, headDim: Int, theta: Double, thetaHW: Double, dtype: DType) {
        func table(_ positions: [Int32], dim: Int, base: Double) -> (MLXArray, MLXArray) {
            let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
            let invFreq = 1 / pow(Float(base), exponents)
            let freqs = MLXArray(positions).asType(.float32).reshaped([positions.count, 1]) * invFreq.reshaped([1, dim / 2])
            let emb = concatenated([freqs, freqs], axis: -1).reshaped([1, 1, positions.count, dim])
            return (cos(emb).asType(dtype), sin(emb).asType(dtype))
        }
        (cosT, sinT) = table(positions.t, dim: headDim / 2, base: theta)
        (cosH, sinH) = table(positions.h, dim: headDim / 4, base: thetaHW)
        (cosW, sinW) = table(positions.w, dim: headDim / 4, base: thetaHW)
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.shape[x.ndim - 1] / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, 0 ..< half]], axis: -1)
    }

    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        x * cos + rotateHalf(x) * sin
    }
}

final class SenseNovaAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SenseNovaRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SenseNovaRMSNorm
    @ModuleInfo(key: "q_norm_hw") var qNormHW: SenseNovaRMSNorm
    @ModuleInfo(key: "k_norm_hw") var kNormHW: SenseNovaRMSNorm

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    init(config: SenseNovaConfig) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = 1 / Float(config.headDim).squareRoot()
        let hidden = config.hiddenSize
        _qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        _qNorm.wrappedValue = SenseNovaRMSNorm(dimensions: headDim / 2, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SenseNovaRMSNorm(dimensions: headDim / 2, eps: config.rmsNormEps)
        _qNormHW.wrappedValue = SenseNovaRMSNorm(dimensions: headDim / 2, eps: config.rmsNormEps)
        _kNormHW.wrappedValue = SenseNovaRMSNorm(dimensions: headDim / 2, eps: config.rmsNormEps)
        super.init()
    }

    /// Projects, normalizes and rotates `x` [B, S, hidden] into heads-first [B, heads, S, headDim].
    private func project(_ projection: Linear, norm: SenseNovaRMSNorm, normHW: SenseNovaRMSNorm, heads: Int, x: MLXArray, rotary: SenseNovaRotary) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let projected = projection(x).reshaped([batch, length, heads, headDim])
        let half = headDim / 2
        let quarter = headDim / 4
        let t = norm(projected[.ellipsis, 0 ..< half]).transposed(0, 2, 1, 3)
        let hw = normHW(projected[.ellipsis, half...]).transposed(0, 2, 1, 3)
        return concatenated([
            SenseNovaRotary.apply(t, cos: rotary.cosT, sin: rotary.sinT),
            SenseNovaRotary.apply(hw[.ellipsis, 0 ..< quarter], cos: rotary.cosH, sin: rotary.sinH),
            SenseNovaRotary.apply(hw[.ellipsis, quarter...], cos: rotary.cosW, sin: rotary.sinW),
        ], axis: -1)
    }

    /// The keys and values `x` leaves for later tokens, [B, kvHeads, S, headDim] each.
    func keysValues(_ x: MLXArray, rotary: SenseNovaRotary) -> (MLXArray, MLXArray) {
        let keys = project(kProj, norm: kNorm, normHW: kNormHW, heads: kvHeads, x: x, rotary: rotary)
        let values = vProj(x).reshaped([x.shape[0], x.shape[1], kvHeads, headDim]).transposed(0, 2, 1, 3)
        return (keys, values)
    }

    /// Attention of `x`'s tokens over `prefix` (keys and values cached earlier, nil for none) and
    /// themselves, with an additive `mask` over all of them (nil: every token sees every other).
    /// Returns the output and `x`'s own keys and values.
    func callAsFunction(
        _ x: MLXArray, rotary: SenseNovaRotary, prefix: (MLXArray, MLXArray)?, mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (batch, length) = (x.shape[0], x.shape[1])
        let queries = project(qProj, norm: qNorm, normHW: qNormHW, heads: heads, x: x, rotary: rotary)
        let (keys, values) = keysValues(x, rotary: rotary)
        var allKeys = keys
        var allValues = values
        if let prefix {
            allKeys = concatenated([prefix.0, keys], axis: 2)
            allValues = concatenated([prefix.1, values], axis: 2)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues, scale: scale, mask: mask.map { .array($0) } ?? .none
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim])
        return (oProj(merged), keys, values)
    }
}

final class SenseNovaMLP: Module {
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

final class SenseNovaLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SenseNovaRMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SenseNovaAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: SenseNovaRMSNorm
    @ModuleInfo(key: "mlp") var mlp: SenseNovaMLP

    init(config: SenseNovaConfig) {
        _inputLayerNorm.wrappedValue = SenseNovaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfAttn.wrappedValue = SenseNovaAttention(config: config)
        _postAttentionLayerNorm.wrappedValue = SenseNovaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = SenseNovaMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        super.init()
    }
}

/// One of the two stacks: its layers and its final norm (`norm` or `norm_mot_gen` in the checkpoint).
public final class SenseNovaStack: Module {
    @ModuleInfo(key: "layers") var layers: [SenseNovaLayer]
    @ModuleInfo(key: "norm") var norm: SenseNovaRMSNorm

    let config: SenseNovaConfig

    public init(config: SenseNovaConfig) {
        self.config = config
        _layers.wrappedValue = (0 ..< config.numLayers).map { _ in SenseNovaLayer(config: config) }
        _norm.wrappedValue = SenseNovaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// The prefix's keys and values, layer by layer, as `DynamicCache` holds them after the
    /// reference's prefix forward: text (and picture) tokens attending under `mask`, the
    /// block-causal mask of their positions. Only the keys and values are kept, so the last layer
    /// stops once it has them.
    public func prefixCache(_ embeds: MLXArray, positions: SenseNovaPositions, mask: MLXArray) -> [(MLXArray, MLXArray)] {
        let rotary = SenseNovaRotary(positions: positions, headDim: config.headDim, theta: config.ropeTheta, thetaHW: config.ropeThetaHW, dtype: embeds.dtype)
        var h = embeds
        var cache: [(MLXArray, MLXArray)] = []
        for (index, layer) in layers.enumerated() {
            let normed = layer.inputLayerNorm(h)
            if index == layers.count - 1 {
                cache.append(layer.selfAttn.keysValues(normed, rotary: rotary))
                break
            }
            let (attended, keys, values) = layer.selfAttn(normed, rotary: rotary, prefix: nil, mask: mask)
            cache.append((keys, values))
            h = h + attended
            h = h + layer.mlp(layer.postAttentionLayerNorm(h))
        }
        return cache
    }

    /// The image tokens through every layer, each attending (without a mask) over the prefix's
    /// cached keys and values and the image's own, then the final norm: [B, N, hidden].
    public func callAsFunction(
        _ embeds: MLXArray, positions: SenseNovaPositions, prefix: [(MLXArray, MLXArray)], afterLayer: (Int) -> Void = { _ in }
    ) -> MLXArray {
        let rotary = SenseNovaRotary(positions: positions, headDim: config.headDim, theta: config.ropeTheta, thetaHW: config.ropeThetaHW, dtype: embeds.dtype)
        var h = embeds
        for (index, layer) in layers.enumerated() {
            let (attended, _, _) = layer.selfAttn(layer.inputLayerNorm(h), rotary: rotary, prefix: prefix[index], mask: nil)
            h = h + attended
            h = h + layer.mlp(layer.postAttentionLayerNorm(h))
            afterLayer(index)
        }
        return norm(h)
    }

    /// `create_block_causal_mask`: a token sees the tokens before it and those at its own text
    /// position (a picture's tokens see each other), additive, [1, 1, S, S] in `dtype`.
    public static func blockCausalMask(_ positions: SenseNovaPositions, dtype: DType) -> MLXArray {
        let count = positions.count
        let t = MLXArray(positions.t)
        let index = MLXArray(0 ..< Int32(count))
        let allowed = (t.reshaped([1, count]) .== t.reshaped([count, 1])) .|| (index.reshaped([1, count]) .<= index.reshaped([count, 1]))
        return MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype).reshaped([1, 1, count, count])
    }

    /// Splits the checkpoint's language-model tensors (keys after `language_model.model.`) into the
    /// two stacks': `_mot_gen` ones to the generation stack without the suffix, the rest to the
    /// understanding stack; `norm_mot_gen` is the generation stack's `norm`.
    public static func split(_ tensors: [String: MLXArray]) -> (understanding: [String: MLXArray], generation: [String: MLXArray]) {
        var understanding: [String: MLXArray] = [:]
        var generation: [String: MLXArray] = [:]
        for (key, value) in tensors where key.hasPrefix("layers.") || key.hasPrefix("norm") {
            if key.contains("_mot_gen") {
                generation[key.replacingOccurrences(of: "_mot_gen", with: "")] = value
            } else {
                understanding[key] = value
            }
        }
        return (understanding, generation)
    }
}
