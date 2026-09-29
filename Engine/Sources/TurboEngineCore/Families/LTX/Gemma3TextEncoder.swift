import Foundation
import MLX
import MLXNN

// LTX-2's text encoder: Gemma 3 12B read for the hidden states of every layer, as dgrauet's
// `GemmaLanguageModel.get_all_hidden_states` runs mlx-lm's `gemma3_text` (the embedding, then
// each of the 48 layers called with one causal-and-padding mask, no sliding window, no cache).
// Module names follow mlx-lm, so `mlx-community/gemma-3-12b-it-4bit` loads unchanged from its
// `language_model.model.` keys; the vision tower in the same files is skipped.

struct Gemma3Config {
    var hiddenSize = 3840
    var intermediateSize = 15360
    var heads = 16
    var kvHeads = 8
    var headDim = 256
    var layers = 48
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 1_000_000
    var ropeLocalBase: Float = 10_000
    /// Global layers only: 1 / the linear scaling factor (8 for the 12B model).
    var ropeGlobalScale: Float = 1
    var queryPreAttnScalar: Float = 256
    var slidingWindowPattern = 6
    var vocabSize = 262_208

    /// The `text_config` of the checkpoint's `config.json`, with mlx-lm's defaults for the rest.
    static func load(folder: URL) throws -> Gemma3Config {
        var config = Gemma3Config()
        let data = try Data(contentsOf: folder.appending(path: "config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return config }
        let text = (json["text_config"] as? [String: Any]) ?? json
        config.hiddenSize = text["hidden_size"] as? Int ?? config.hiddenSize
        config.intermediateSize = text["intermediate_size"] as? Int ?? config.intermediateSize
        config.heads = text["num_attention_heads"] as? Int ?? config.heads
        config.kvHeads = text["num_key_value_heads"] as? Int ?? config.kvHeads
        config.headDim = text["head_dim"] as? Int ?? config.headDim
        config.layers = text["num_hidden_layers"] as? Int ?? config.layers
        if let value = text["rms_norm_eps"] as? Double { config.rmsNormEps = Float(value) }
        if let value = text["rope_theta"] as? Double { config.ropeTheta = Float(value) }
        if let value = text["rope_local_base_freq"] as? Double { config.ropeLocalBase = Float(value) }
        if let value = text["query_pre_attn_scalar"] as? Double { config.queryPreAttnScalar = Float(value) }
        config.slidingWindowPattern = text["sliding_window_pattern"] as? Int ?? config.slidingWindowPattern
        if let scaling = text["rope_scaling"] as? [String: Any], (scaling["rope_type"] as? String) == "linear",
           let factor = scaling["factor"] as? Double {
            config.ropeGlobalScale = Float(1 / factor)
        }
        config.vocabSize = (json["vocab_size"] as? Int) ?? (text["vocab_size"] as? Int) ?? config.vocabSize
        return config
    }
}

/// `mx.fast.rms_norm(x, 1 + weight, eps)`: Gemma stores the weight as an offset from one.
final class Gemma3RMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

final class Gemma3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma3RMSNorm

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let ropeBase: Float
    let ropeScale: Float

    init(config: Gemma3Config, layer: Int) {
        heads = config.heads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = 1 / config.queryPreAttnScalar.squareRoot()
        // Every sliding_window_pattern-th layer is global: the long base and its scaling.
        let isSliding = (layer + 1) % config.slidingWindowPattern != 0
        ropeBase = isSliding ? config.ropeLocalBase : config.ropeTheta
        ropeScale = isSliding ? 1 : config.ropeGlobalScale
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = Gemma3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Gemma3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// `mask` is additive, [B, 1, T, T]. Grouped-query attention is left to the fused kernel.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        var q = qProj(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        q = qNorm(q)
        k = kNorm(k)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeBase, scale: ropeScale, offset: 0)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeBase, scale: ropeScale, offset: 0)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }
}

final class Gemma3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma3Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma3RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma3RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma3RMSNorm

    init(config: Gemma3Config, layer: Int) {
        _selfAttn.wrappedValue = Gemma3Attention(config: config, layer: layer)
        _mlp.wrappedValue = Gemma3MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// mlx-lm's `clip_residual` is a plain sum outside float16.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let attended = selfAttn(inputLayerNorm(x), mask: mask)
        let h = x + postAttentionLayerNorm(attended)
        let fed = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(fed)
    }
}

/// `gemma3_text.Gemma3Model` (`language_model.model` in the checkpoint).
final class Gemma3TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma3RMSNorm

    let config: Gemma3Config

    init(config: Gemma3Config) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layers).map { Gemma3DecoderLayer(config: config, layer: $0) }
        _norm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// The embedding (scaled by √hidden, in bfloat16 as mlx-lm scales it) and the output of every
    /// layer: 49 arrays [B, T, hidden]. `attentionMask` is [B, T], 1 on real tokens; padding and
    /// the future are masked with −1e9 in bfloat16. Each layer is evaluated as it finishes, which
    /// keeps every Metal command buffer short.
    func allHiddenStates(tokens: MLXArray, attentionMask: MLXArray) -> [MLXArray] {
        var h = embedTokens(tokens)
        let scale = MLXArray(Float(config.hiddenSize).squareRoot()).asType(.bfloat16).asType(h.dtype)
        h = h * scale
        var states = [h]

        let length = tokens.shape[1]
        let blocked = MLXArray(Float(-1e9)).asType(.bfloat16)
        let causal = triu(MLXArray.full([length, length], values: blocked), k: 1)
        let padding = (1 - attentionMask.expandedDimensions(axes: [1, 2]).asType(.bfloat16)) * blocked
        let mask = causal.expandedDimensions(axes: [0, 1]) + padding

        for layer in layers {
            h = layer(h, mask: mask)
            states.append(h)
            eval(h)
        }
        return states
    }

    /// The checkpoint keys this model reads, without their `language_model.model.` prefix.
    static func weights(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        let prefix = "language_model.model."
        var weights: [String: MLXArray] = [:]
        for (key, value) in tensors where key.hasPrefix(prefix) {
            weights[String(key.dropFirst(prefix.count))] = value
        }
        return weights
    }
}

/// Gemma's tokenizer as the port calls it: `encode(prompt.strip())` (with the leading `<bos>`),
/// the last `maxLength` tokens kept, left-padded with the pad token.
public final class Gemma3Prompter {
    private let tokenizer: BPETokenizer
    private let padTokenId: Int
    let maxLength: Int

    public init(folder: URL, maxLength: Int) throws {
        guard FileManager.default.fileExists(atPath: folder.appending(path: "tokenizer.json").path) else {
            throw Qwen3Prompter.PromptError.noTokenizer(folder)
        }
        tokenizer = try BPETokenizer(folder: folder)
        padTokenId = tokenizer.id(of: "<pad>") ?? 0
        self.maxLength = maxLength
    }

    public func tokenIds(_ prompt: String) -> [Int] {
        var ids = tokenizer.encode(prompt.trimmingCharacters(in: .whitespacesAndNewlines), addSpecialTokens: true)
        if ids.count > maxLength { ids = Array(ids.suffix(maxLength)) }
        return ids
    }

    /// Token ids and attention mask, both [1, maxLength] int32.
    func tokenize(_ prompt: String) -> (tokens: MLXArray, mask: MLXArray) {
        let ids = tokenIds(prompt)
        let padding = maxLength - ids.count
        let padded = Array(repeating: padTokenId, count: padding) + ids
        let mask = Array(repeating: Int32(0), count: padding) + Array(repeating: Int32(1), count: ids.count)
        return (MLXArray(padded.map { Int32($0) }, [1, maxLength]), MLXArray(mask, [1, maxLength]))
    }
}
