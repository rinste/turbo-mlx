import Foundation
import MLX
import MLXNN

// MiniMax Music 3's two language models, as mlx-audio runs them: the global one, Qwen3-8B with its embeddings and head
// adapted to music tokens (mlx-lm's `qwen3.py`: fast RMS norms, rotary embedding and attention, a growing key-value
// cache), and the local RVQ depth decoder (`depth.py`). Module and parameter names follow the pack's keys
// (`language_model.…`, `rvq_depth_decoder.…`).

/// mlx-lm's `KVCache`: keys and values in buffers grown 256 positions at a time and written in place.
public final class Music3KVCache {
    static let step = 256
    public private(set) var offset = 0
    private var keys: MLXArray?
    private var values: MLXArray?

    public init() {}

    /// Appends [B, H, L, D] keys and values; the keys and values so far.
    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let count = newKeys.dim(2)
        if keys == nil || previous + count > keys!.dim(2) {
            let length = (Self.step + count - 1) / Self.step * Self.step
            let grownKeys = MLXArray.zeros([newKeys.dim(0), newKeys.dim(1), length, newKeys.dim(3)], dtype: newKeys.dtype)
            let grownValues = MLXArray.zeros([newValues.dim(0), newValues.dim(1), length, newValues.dim(3)], dtype: newValues.dtype)
            if var oldKeys = keys, var oldValues = values {
                if previous % Self.step != 0 {
                    oldKeys = oldKeys[.ellipsis, ..<previous, 0...]
                    oldValues = oldValues[.ellipsis, ..<previous, 0...]
                }
                keys = concatenated([oldKeys, grownKeys], axis: 2)
                values = concatenated([oldValues, grownValues], axis: 2)
            } else {
                keys = grownKeys
                values = grownValues
            }
        }
        offset += count
        keys![.ellipsis, previous ..< offset, 0...] = newKeys
        values![.ellipsis, previous ..< offset, 0...] = newValues
        return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
    }
}

public final class Music3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let heads: Int
    let kvHeads: Int
    let scale: Float
    let rope: RoPE

    init(config: Music3Config) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        let dim = config.hiddenSize, headDim = config.headDim
        scale = Float(pow(Double(headDim), -0.5))
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cache: Music3KVCache) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qNorm(qProj(x).reshaped(batch, length, heads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)
        queries = rope(queries, offset: cache.offset)
        keys = rope(keys, offset: cache.offset)
        let (allKeys, allValues) = cache.update(keys: keys, values: values)
        // A causal mask for the prompt; a single new position sees everything before it.
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues, scale: scale, mask: length > 1 ? .causal : .none)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

public final class Music3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dim: Int, hidden: Int) {
        _gateProj.wrappedValue = Linear(dim, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, dim, bias: false)
        _upProj.wrappedValue = Linear(dim, hidden, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

public final class Music3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Music3Attention
    @ModuleInfo(key: "mlp") var mlp: Music3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: Music3Config) {
        _selfAttn.wrappedValue = Music3Attention(config: config)
        _mlp.wrappedValue = Music3MLP(dim: config.hiddenSize, hidden: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cache: Music3KVCache) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class Music3Qwen3: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Music3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: Music3Config) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in Music3DecoderLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// The normalized last hidden states of `embeddings` [B, L, hidden], the caches extended by L positions.
    public func callAsFunction(_ embeddings: MLXArray, caches: [Music3KVCache]) -> MLXArray {
        var h = embeddings
        for (layer, cache) in zip(layers, caches) { h = layer(h, cache: cache) }
        return norm(h)
    }
}

/// The global language model: `language_model.model` and its head.
public final class Music3LanguageModel: Module {
    @ModuleInfo(key: "model") public var model: Music3Qwen3
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    let config: Music3Config

    init(config: Music3Config) {
        self.config = config
        _model.wrappedValue = Music3Qwen3(config: config)
        _lmHead.wrappedValue = config.tieWordEmbeddings ? nil : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    func makeCaches() -> [Music3KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in Music3KVCache() }
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(hidden) }
        return model.embedTokens.asLinear(hidden)
    }
}

// MARK: The depth decoder

public final class Music3DepthAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int
    let scale: Float

    init(dim: Int, heads: Int) {
        self.heads = heads
        scale = Float(1.0 / Double(dim / heads).squareRoot())
        _toQ.wrappedValue = Linear(dim, dim, bias: false)
        _toK.wrappedValue = Linear(dim, dim, bias: false)
        _toV.wrappedValue = Linear(dim, dim, bias: false)
        _toOut.wrappedValue = Linear(dim, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q = toQ(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        let k = toK(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal)
        return toOut(y.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

public final class Music3DepthBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "attn") var attn: Music3DepthAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Music3Config) {
        let dim = config.hiddenSize
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        _attn.wrappedValue = Music3DepthAttention(dim: dim, heads: config.depthNumHeads)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        _gateProj.wrappedValue = Linear(dim, config.depthIntermediateSize, bias: false)
        _upProj.wrappedValue = Linear(dim, config.depthIntermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.depthIntermediateSize, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attn(inputLayerNorm(x))
        let normalized = postAttentionLayerNorm(h)
        return h + downProj(silu(gateProj(normalized)) * upProj(normalized))
    }
}

/// The local model: from the global model's last hidden state and the frame's semantic code, the seven residual codes,
/// one after the other, each read from its own head.
public final class Music3DepthDecoder: Module {
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
    @ModuleInfo(key: "projection") var projection: Linear
    @ModuleInfo(key: "pos_embedding") var positionEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [Music3DepthBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "audio_heads") var audioHeads: [Linear]

    init(config: Music3Config) {
        let residual = config.residualCodebooks
        _audioEmbeddings.wrappedValue = Embedding(embeddingCount: config.audioVocabSize * residual, dimensions: config.hiddenSize)
        _projection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: config.depthMaxPositionEmbeddings, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.depthNumLayers).map { _ in Music3DepthBlock(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _audioHeads.wrappedValue = (0 ..< residual).map { _ in Linear(config.hiddenSize, config.audioVocabSize, bias: false) }
        super.init()
    }

    /// The normalized hidden states of the sequence [B, L, hidden] (L at most 8), positions added.
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input + positionEmbedding(MLXArray(Int32(0) ..< Int32(input.dim(1))))
        for layer in layers { hidden = layer(hidden) }
        return norm(hidden)
    }
}
