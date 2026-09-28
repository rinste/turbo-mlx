import Foundation
import MLX
import MLXNN

// Qwen-Image's text encoder: the language model of Qwen2.5-VL 7B, after mflux's
// `qwen_text_encoder/*`, with the vision tower for an edit's picture (Qwen25VisionTower.swift).
// mflux numbers every token's position alike on the three rotary axes, the picture's too, so the
// multimodal rotary embedding reduces to the plain one. Names follow mflux, whose checkpoints keep
// it unquantized in bf16.

final class Qwen25Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    init(config: QwenImageConfig.TextEncoder) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = 1 / Float(config.headDim).squareRoot()
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        super.init()
    }

    /// `mask` is additive, [B, 1, S, S] float32; `cos`/`sin` are [B, S, headDim] float32.
    func callAsFunction(_ x: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        var q = qProj(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)

        // HF-style rotary over the full head in float32, cast back.
        let cosB = cos.expandedDimensions(axis: 1)
        let sinB = sin.expandedDimensions(axis: 1)
        let q32 = q.asType(.float32)
        let k32 = k.asType(.float32)
        q = (q32 * cosB + Self.rotateHalf(q32) * sinB).asType(q.dtype)
        k = (k32 * cosB + Self.rotateHalf(k32) * sinB).asType(k.dtype)

        if kvHeads != heads {
            let repeats = heads / kvHeads
            k = repeated(k, count: repeats, axis: 1)
            v = repeated(v, count: repeats, axis: 1)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask.asType(q.dtype)
        )
        return oProj(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.shape[x.ndim - 1] / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, 0 ..< half]], axis: -1)
    }
}

final class Qwen25DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Qwen3RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen25Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Qwen3RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP

    init(config: QwenImageConfig.TextEncoder) {
        _inputLayerNorm.wrappedValue = Qwen3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfAttn.wrappedValue = Qwen25Attention(config: config)
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

/// mflux's `QwenEncoder`: embeddings, the decoder layers and the final norm, and for an edit the
/// vision tower whose tokens replace the picture's placeholders.
final class Qwen25Encoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen25DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Qwen3RMSNorm
    @ModuleInfo(key: "visual") var visual: Qwen25VisionTower?

    let config: QwenImageConfig.TextEncoder

    init(config: QwenImageConfig.TextEncoder, vision: QwenImageConfig.Vision? = nil) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in Qwen25DecoderLayer(config: config) }
        _norm.wrappedValue = Qwen3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _visual.wrappedValue = vision.map { Qwen25VisionTower($0, outDim: config.hiddenSize) }
        super.init()
    }

    /// The final-norm hidden states [B, S, hidden] in the weights' dtype, or in float32 when a
    /// picture's tokens (`imageEmbeds`, float32) are placed in the sequence.
    func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray, imageEmbeds: MLXArray? = nil) -> MLXArray {
        let (batch, length) = (inputIds.shape[0], inputIds.shape[1])
        var h = embedTokens(inputIds)
        if let imageEmbeds {
            h = Self.placing(imageEmbeds, in: h, inputIds: inputIds, tokenId: config.imageTokenId)
        }

        // Padding (keys of padded tokens) plus causality, additive in float32.
        let padding = MLX.where(attentionMask .== 1, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(.float32)
            .reshaped([batch, 1, 1, length])
        let positions = MLXArray(0 ..< Int32(length))
        let causalBool = positions.expandedDimensions(axis: 0) .> positions.expandedDimensions(axis: 1)
        let causal = MLX.where(causalBool, MLXArray(-Float.infinity), MLXArray(Float(0)))
            .asType(.float32)
            .reshaped([1, 1, length, length])
        let mask = broadcast(causal, to: [batch, 1, length, length]) + padding

        let (cos, sin) = rotary(batch: batch, length: length, dtype: h.dtype)
        for layer in layers {
            h = layer(h, mask: mask, cos: cos, sin: sin)
        }
        return norm(h)
    }

    /// As mflux's `QwenEncoder` does it: the placeholder tokens' embeddings replaced, in order, by
    /// the picture's rows, all or nothing (only when there are rows enough). The reference stacks
    /// the rows, so float32 picture rows make the whole sequence float32; concatenating does too.
    static func placing(_ image: MLXArray, in embeds: MLXArray, inputIds: MLXArray, tokenId: Int) -> MLXArray {
        let ids = inputIds.reshaped([-1]).asArray(Int32.self)
        let placeholders = ids.filter { Int($0) == tokenId }.count
        guard placeholders > 0, image.shape[0] >= placeholders else { return embeds }
        let hidden = embeds.shape[embeds.ndim - 1]
        let rows = embeds.reshaped([-1, hidden])
        var next = 0
        let index = ids.enumerated().map { position, id -> Int32 in
            guard Int(id) == tokenId, next < image.shape[0] else { return Int32(position) }
            next += 1
            return Int32(rows.shape[0] + next - 1)
        }
        return concatenated([rows, image], axis: 0)[MLXArray(index)].reshaped(embeds.shape)
    }

    /// `QwenRotaryEmbedding` on text positions 0..<S (the same on all three axes, so the
    /// multimodal sections reduce to the full rotation): cos and sin [B, S, headDim], computed in
    /// float32, rounded to the activations' dtype as the reference does, then used in float32.
    private func rotary(batch: Int, length: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let dim = config.headDim
        let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let invFreq = 1 / pow(Float(config.ropeTheta), exponents)
        let positions = MLXArray((0 ..< length).map { Float($0) }).reshaped([1, length, 1])
        let freqs = positions * invFreq.reshaped([1, 1, dim / 2])
        let emb = broadcast(concatenated([freqs, freqs], axis: -1), to: [batch, length, dim])
        return (cos(emb).asType(dtype).asType(.float32), sin(emb).asType(dtype).asType(.float32))
    }
}

/// mflux's `QwenTextEncoder`: the encoder, then the real tokens after the template's system
/// prompt, as the transformer reads them.
public final class Qwen25TextEncoder: Module {
    @ModuleInfo(key: "encoder") var encoder: Qwen25Encoder

    public let config: QwenImageConfig.TextEncoder

    public init(config: QwenImageConfig.TextEncoder, vision: QwenImageConfig.Vision? = nil) {
        self.config = config
        _encoder.wrappedValue = Qwen25Encoder(config: config, vision: vision)
        super.init()
    }

    /// Keys of the checkpoint this module has no parameter for: the rotary tables (computed) and
    /// the vision tower (never loaded for text-to-image).
    public static func ignoresKey(_ key: String) -> Bool {
        key.contains("rotary_emb.") || key.hasPrefix("encoder.visual.")
    }

    /// The same for an edit, which loads the vision tower (its rotary frequencies are computed too).
    public static func ignoresKeyWithVision(_ key: String) -> Bool {
        key.contains("rotary_emb.") || key.contains("rotary_pos_emb.")
    }

    /// The picture's tokens: `pixelValues` [N, 1176] of the pictures `grids` describes → [N / 4, hidden].
    public func imageEmbeds(pixelValues: MLXArray, grids: [QwenVisionGrid]) -> MLXArray {
        guard let visual = encoder.visual else { preconditionFailure("this text encoder has no vision tower") }
        return visual(pixelValues, grids: grids)
    }

    /// An edit's prompt, as `QwenVisionLanguageEncoder` and `QwenImageEdit` make it: the encoder
    /// over the template with the picture's tokens in place, the states after the 64-token system
    /// prompt, cast to float16. `imageEmbeds` from `imageEmbeds(pixelValues:grids:)`.
    public func editEmbeds(inputIds: MLXArray, imageEmbeds: MLXArray) -> MLXArray {
        let mask = MLXArray.ones(inputIds.shape, dtype: .int32)
        let hidden = encoder(inputIds, attentionMask: mask, imageEmbeds: imageEmbeds)
        let kept = hidden.shape[1] > config.dropIndex ? hidden[0..., config.dropIndex..., 0...] : hidden
        return kept.asType(.float16)
    }

    /// `inputIds` [1, S] of one prompt (no padding) → its embeddings [1, S − dropIndex, hidden]
    /// in bf16, every token real.
    public func promptEmbeds(inputIds: MLXArray) -> MLXArray {
        let mask = MLXArray.ones(inputIds.shape, dtype: .int32)
        let hidden = encoder(inputIds, attentionMask: mask)
        let kept = hidden[0..., config.dropIndex..., 0...]
        return kept.asType(modelPrecision)
    }
}
