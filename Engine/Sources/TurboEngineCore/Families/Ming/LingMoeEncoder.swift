import Foundation
import MLX
import MLXNN

// Ming-Image's text encoder: Ling-mini-2.0 (`bailing_moe_v2`), after mflux's
// `ling_moe_encoder.py`. Only the pieces a text-to-image prompt pass touches are built: the
// Qwen2.5 vision tower, the language head and the audio router are never loaded. Names follow
// mflux, whose saved checkpoints keep the experts stacked for gathered matrix multiplications.

/// `video_rope`: 3D (t, h, w) positions over the first 64 of each head's 128 dims (rotate-half
/// layout). Of the 32 frequencies, the first 24 alternate between the h (even) and w (odd)
/// position and the last 8 follow t.
enum LingRope {
    /// `positionIds` [3, L] int32 → cos and sin [L, ropeDim] float32.
    static func cosSin(positionIds: MLXArray, ropeDim: Int, theta: Float) -> (MLXArray, MLXArray) {
        let half = ropeDim / 2
        let exponents = MLXArray(stride(from: 0, to: ropeDim, by: 2).map { Float($0) / Float(ropeDim) })
        let invFreq = 1 / pow(theta, exponents)
        let axes = MLXArray((0 ..< half).map { f -> Int32 in f >= 24 ? 0 : (f % 2 == 0 ? 1 : 2) })
        let positions = positionIds.asType(.float32).take(axes, axis: 0)          // [half, L]
        let freqs = positions.transposed() * invFreq.expandedDimensions(axis: 0)   // [L, half]
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb), sin(emb))
    }

    /// `x` [B, H, L, headDim]; cos and sin [L, ropeDim], cast to x's dtype.
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray, ropeDim: Int) -> MLXArray {
        let cos = cos.asType(x.dtype)
        let sin = sin.asType(x.dtype)
        let rotated = x[.ellipsis, 0 ..< ropeDim]
        let passed = x[.ellipsis, ropeDim...]
        let half = ropeDim / 2
        let rotatedHalf = concatenated([-rotated[.ellipsis, half...], rotated[.ellipsis, 0 ..< half]], axis: -1)
        return concatenated([rotated * cos + rotatedHalf * sin, passed], axis: -1)
    }
}

final class LingAttention: Module {
    @ModuleInfo(key: "query_key_value") var queryKeyValue: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "dense") var dense: Linear

    let config: MingConfig.Encoder

    init(config: MingConfig.Encoder) {
        self.config = config
        _queryKeyValue.wrappedValue = Linear(config.hiddenSize, (config.numHeads + 2 * config.numKvHeads) * config.headDim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsEps)
        _dense.wrappedValue = Linear(config.numHeads * config.headDim, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, causalMask: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let (heads, kvHeads, headDim) = (config.numHeads, config.numKvHeads, config.headDim)
        let qkv = queryKeyValue(x).reshaped([batch, length, heads + 2 * kvHeads, headDim])
        var q = qNorm(qkv[0..., 0..., 0 ..< heads]).transposed(0, 2, 1, 3)
        var k = kNorm(qkv[0..., 0..., heads ..< (heads + kvHeads)]).transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., (heads + kvHeads)...].transposed(0, 2, 1, 3)
        q = LingRope.apply(q, cos: cos, sin: sin, ropeDim: config.ropeDim)
        k = LingRope.apply(k, cos: cos, sin: sin, ropeDim: config.ropeDim)
        // Grouped-query attention (the kernel reads each key head for its group), causal.
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: causalMask.asType(q.dtype)
        )
        return dense(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }
}

final class LingMLP: Module {
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

/// Sigmoid router with expert bias and group-limited top-k (8 groups, the best 4 by top-2 sum).
/// The bias only steers selection; the mixing weights come from the unbiased scores. Upstream runs
/// this under torch's bf16 autocast, so logits, scores and `scores + expert_bias` are bf16 (the
/// sigmoid evaluated in float32 and rounded once) while the group scores and the mixing weights
/// are float32, and torch's top-k resolves ties towards the lower index. mflux reproduces all of
/// that, and so does this: routing in plain float32 picks noticeably different experts.
final class LingGate: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    let config: MingConfig.Encoder

    init(config: MingConfig.Encoder) {
        self.config = config
        _weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize])
        _expertBias.wrappedValue = MLXArray.zeros([config.numExperts])
        super.init()
    }

    /// `x` [N, hidden] → the chosen experts [N, topK] and their weights [N, topK] float32.
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let dtype = DType.bfloat16
        let logits = matmul(x.asType(dtype), weight.asType(dtype).transposed())
        let scores = sigmoid(logits.asType(.float32)).asType(dtype)
        let routing = scores + expertBias.asType(dtype)
        let n = routing.shape[0]
        let perGroup = config.numExperts / config.nGroup
        let grouped = routing.reshaped([n, config.nGroup, perGroup])
        let groupScores = top(grouped, k: 2, axis: -1).asType(.float32).sum(axis: -1)
        let groupIndex = Self.topKLowIndexFirst(groupScores, k: config.topkGroup, tieEps: 1e-3)
        let groupMask = putAlong(MLXArray.zeros([n, config.nGroup]), groupIndex, values: MLXArray(Float(1)), axis: -1)
        let expertMask = repeated(groupMask.expandedDimensions(axis: -1), count: perGroup, axis: -1).reshaped([n, config.numExperts])
        let masked = MLX.where(expertMask .> 0, routing.asType(.float32), MLXArray(-Float.infinity))
        let topkIndex = Self.topKLowIndexFirst(masked, k: config.topK, tieEps: 1e-5)
        var weights = takeAlong(scores, topkIndex, axis: -1).asType(.float32)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20) * config.routedScaling
        return (topkIndex, weights)
    }

    /// The k largest values' indices, ties resolved towards the lower index: the values are bf16
    /// routing scores near 17 (spacing 0.125) or float32 sums of two of them, so subtracting
    /// `index · tieEps` orders ties by index without reordering distinct values.
    static func topKLowIndexFirst(_ values: MLXArray, k: Int, tieEps: Float) -> MLXArray {
        let count = values.shape[values.ndim - 1]
        let key = values.asType(.float32) - MLXArray((0 ..< count).map { Float($0) }) * tieEps
        return argSort(-key, axis: -1)[.ellipsis, 0 ..< k]
    }
}

/// MultiRouter MoE: text tokens use `gate`, image-patch tokens (Ming's learned query tokens) use
/// `image_gate`; one shared expert is always on.
final class LingSparseMoe: Module {
    @ModuleInfo(key: "gate") var gate: LingGate
    @ModuleInfo(key: "image_gate") var imageGate: LingGate
    @ModuleInfo(key: "switch_mlp") var switchMlp: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: LingMLP

    init(config: MingConfig.Encoder) {
        _gate.wrappedValue = LingGate(config: config)
        _imageGate.wrappedValue = LingGate(config: config)
        _switchMlp.wrappedValue = SwitchGLU(inputDims: config.hiddenSize, hiddenDims: config.moeIntermediate, numExperts: config.numExperts)
        _sharedExperts.wrappedValue = LingMLP(hiddenSize: config.hiddenSize, intermediateSize: config.moeIntermediate)
        super.init()
    }

    /// `imageMask` [B, L] bool marks the query tokens.
    func callAsFunction(_ x: MLXArray, imageMask: MLXArray?) -> MLXArray {
        let dim = x.shape[x.ndim - 1]
        let flat = x.reshaped([-1, dim])
        var (indices, weights) = gate(flat)
        if let imageMask {
            let (imageIndices, imageWeights) = imageGate(flat)
            let mask = imageMask.reshaped([-1, 1])
            indices = MLX.where(mask, imageIndices, indices)
            weights = MLX.where(mask, imageWeights, weights)
        }
        let expertOut = switchMlp(flat, indices)  // [N, topK, dim]
        // Weighted and summed in float32 (the weights' dtype), as upstream's moe_infer does.
        let y = (expertOut.asType(.float32) * weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(x.dtype)
        return y.reshaped(x.shape) + sharedExperts(x)
    }
}

final class LingDecoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: LingAttention
    /// A `LingMLP` for the first layers, a `LingSparseMoe` for the rest.
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: MingConfig.Encoder, index: Int) {
        _attention.wrappedValue = LingAttention(config: config)
        _mlp.wrappedValue = index < config.firstKDense
            ? LingMLP(hiddenSize: config.hiddenSize, intermediateSize: config.denseIntermediate)
            : LingSparseMoe(config: config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, causalMask: MLXArray, imageMask: MLXArray?) -> MLXArray {
        let x = x + attention(inputLayerNorm(x), cos: cos, sin: sin, causalMask: causalMask)
        let h = postAttentionLayerNorm(x)
        if let dense = mlp as? LingMLP {
            return x + dense(h)
        }
        return x + (mlp as! LingSparseMoe)(h, imageMask: imageMask)
    }
}

public final class LingMoeEncoder: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [LingDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public let config: MingConfig.Encoder

    public init(config: MingConfig.Encoder) {
        self.config = config
        _wordEmbeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numLayers).map { LingDecoderLayer(config: config, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEps)
        super.init()
    }

    /// HF-style hidden states by index: `i` is the input to layer i (0 = embeddings) and
    /// `numLayers` the final-norm output; only the `outputLayers` asked for are kept.
    public func callAsFunction(inputsEmbeds: MLXArray, positionIds: MLXArray, imageMask: MLXArray?, outputLayers: Set<Int>) -> [Int: MLXArray] {
        let (cos, sin) = LingRope.cosSin(positionIds: positionIds, ropeDim: config.ropeDim, theta: config.ropeTheta)
        let length = inputsEmbeds.shape[1]
        let positions = MLXArray(0 ..< Int32(length))
        let future = positions.expandedDimensions(axis: 0) .> positions.expandedDimensions(axis: 1)
        let causalMask = MLX.where(future, MLXArray(-Float.infinity), MLXArray(Float(0)))
        var out: [Int: MLXArray] = [:]
        var h = inputsEmbeds
        for (index, layer) in layers.enumerated() {
            if outputLayers.contains(index) { out[index] = h }
            h = layer(h, cos: cos, sin: sin, causalMask: causalMask, imageMask: imageMask)
        }
        if outputLayers.contains(config.numLayers) { out[config.numLayers] = norm(h) }
        return out
    }
}
