import Foundation
import MLX
import MLXNN

// From Gemma's 49 hidden states to the transformer's two text contexts: dgrauet's
// `GemmaFeaturesExtractorV2` (per-token RMS over the stacked layers, one projection per modality)
// and its `Embeddings1DConnector`s (learnable registers in place of the padding, eight gated
// attention blocks with RoPE). Weights: the pack's `connector.safetensors`, all bfloat16. The
// checkpoint wraps two layers in lists (`attn1.to_out.0`, `ff.net.0.proj` / `ff.net.2`), which the
// loader renames to plain properties.

/// `mx.fast.rms_norm(x, None, eps)`: RMS normalization without a weight.
func unweightedRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.shape[x.ndim - 1]], dtype: x.dtype), eps: eps)
}

final class LTXConnectorAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "to_gate_logits") var toGateLogits: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let heads: Int
    let headDim: Int
    let scale: Float

    init(dim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        scale = 1 / Float(headDim).squareRoot()
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: true)
        _toK.wrappedValue = Linear(dim, inner, bias: true)
        _toV.wrappedValue = Linear(dim, inner, bias: true)
        _toOut.wrappedValue = Linear(inner, dim, bias: true)
        _toGateLogits.wrappedValue = Linear(dim, heads, bias: true)
        // nn.RMSNorm's default epsilon, not the transformer's.
        _qNorm.wrappedValue = RMSNorm(dimensions: inner, eps: 1e-5)
        _kNorm.wrappedValue = RMSNorm(dimensions: inner, eps: 1e-5)
        super.init()
    }

    /// Plain softmax attention (the reference does not use the fused kernel here), gated per head
    /// by `2·sigmoid(logits)`.
    func callAsFunction(_ x: MLXArray, rope: LTXRope) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let q = qNorm(toQ(x)).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let k = kNorm(toK(x)).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let qr = rope.apply(q)
        let kr = rope.apply(k)
        let weights = softmax(matmul(qr, kr.transposed(0, 1, 3, 2)) * scale, axis: -1)
        var out = matmul(weights, v)
        let gate = 2 * sigmoid(toGateLogits(x))
        out = out * gate.transposed(0, 2, 1).expandedDimensions(axis: -1)
        return toOut(out.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }
}

final class LTXConnectorFeedForward: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dim: Int, mult: Int) {
        _projIn.wrappedValue = Linear(dim, dim * mult, bias: true)
        _projOut.wrappedValue = Linear(dim * mult, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projOut(geluApproximate(projIn(x)))
    }
}

final class LTXConnectorBlock: Module {
    @ModuleInfo(key: "attn1") var attn1: LTXConnectorAttention
    @ModuleInfo(key: "ff") var ff: LTXConnectorFeedForward

    init(dim: Int, heads: Int, headDim: Int, ffMult: Int) {
        _attn1.wrappedValue = LTXConnectorAttention(dim: dim, heads: heads, headDim: headDim)
        _ff.wrappedValue = LTXConnectorFeedForward(dim: dim, mult: ffMult)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: LTXRope) -> MLXArray {
        let h = x + attn1(unweightedRMSNorm(x, eps: 1e-6), rope: rope)
        return h + ff(unweightedRMSNorm(h, eps: 1e-6))
    }
}

final class LTXEmbeddingsConnector: Module {
    @ParameterInfo(key: "learnable_registers") var learnableRegisters: MLXArray
    @ModuleInfo(key: "transformer_1d_blocks") var blocks: [LTXConnectorBlock]

    let dim: Int
    let heads: Int
    let maxPosition: Int
    let doublePrecisionRope: Bool

    init(dim: Int, heads: Int, headDim: Int, layers: Int, registers: Int, ffMult: Int, maxPosition: Int, doublePrecisionRope: Bool) {
        self.dim = dim
        self.heads = heads
        self.maxPosition = maxPosition
        self.doublePrecisionRope = doublePrecisionRope
        _learnableRegisters.wrappedValue = MLXArray.zeros([registers, dim])
        _blocks.wrappedValue = (0 ..< layers).map { _ in LTXConnectorBlock(dim: dim, heads: heads, headDim: headDim, ffMult: ffMult) }
        super.init()
    }

    /// `hidden` [B, T, dim] with the left-padded `validTokens` last: the valid tokens move to the
    /// front and the registers, tiled, fill the rest; every position then attends to every other.
    func callAsFunction(_ hidden: MLXArray, validTokens: Int) -> MLXArray {
        let (batch, length) = (hidden.shape[0], hidden.shape[1])
        let registers = learnableRegisters.shape[0]
        let tiledRegisters = tiled(learnableRegisters, repetitions: [length / registers, 1])

        var rows: [MLXArray] = []
        for b in 0 ..< batch {
            let valid = hidden[b, (length - validTokens)..., 0...]
            let adjusted = validTokens < length
                ? concatenated([valid, MLXArray.zeros([length - validTokens, dim], dtype: valid.dtype)], axis: 0)
                : valid
            let flipped = concatenated([
                MLXArray.ones([validTokens, 1], dtype: adjusted.dtype),
                MLXArray.zeros([length - validTokens, 1], dtype: adjusted.dtype),
            ], axis: 0)
            rows.append(flipped * adjusted + (1 - flipped) * tiledRegisters)
        }
        var h = stacked(rows, axis: 0)

        let positions = MLXArray(0 ..< length).asType(.float32).reshaped([1, length, 1])
        let rope = LTXRope(positions: positions, innerDim: dim, heads: heads, theta: 10000, maxPositions: [maxPosition],
                           doublePrecision: doublePrecisionRope)
        for block in blocks {
            h = block(h, rope: rope)
            eval(h)
        }
        return unweightedRMSNorm(h, eps: 1e-6)
    }
}

final class LTXTextEmbeddingProjection: Module {
    @ModuleInfo(key: "video_aggregate_embed") var videoAggregateEmbed: Linear
    @ModuleInfo(key: "audio_aggregate_embed") var audioAggregateEmbed: Linear

    let embeddingDim: Int

    init(inputDim: Int, videoDim: Int, audioDim: Int, embeddingDim: Int) {
        self.embeddingDim = embeddingDim
        _videoAggregateEmbed.wrappedValue = Linear(inputDim, videoDim, bias: true)
        _audioAggregateEmbed.wrappedValue = Linear(inputDim, audioDim, bias: true)
        super.init()
    }

    /// Each projection sees the features rescaled by √(its width / Gemma's).
    func callAsFunction(_ features: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        let videoScale = Float((Double(videoAggregateEmbed.weight.shape[0]) / Double(embeddingDim)).squareRoot())
        let video = videoAggregateEmbed(features * videoScale)
        eval(video)
        let audioScale = Float((Double(audioAggregateEmbed.weight.shape[0]) / Double(embeddingDim)).squareRoot())
        let audio = audioAggregateEmbed(features * audioScale)
        eval(audio)
        return (video, audio)
    }
}

/// `TextEncoderConnector`: the projection and the two connectors (`connector.` in the pack).
final class LTXTextConnector: Module {
    @ModuleInfo(key: "text_embedding_projection") var projection: LTXTextEmbeddingProjection
    @ModuleInfo(key: "video_embeddings_connector") var videoConnector: LTXEmbeddingsConnector
    @ModuleInfo(key: "audio_embeddings_connector") var audioConnector: LTXEmbeddingsConnector

    init(config: LTXConfig) {
        let c = config.connector
        let t = config.transformer
        _projection.wrappedValue = LTXTextEmbeddingProjection(
            inputDim: c.gemmaLayers * c.captionChannels, videoDim: t.videoDim, audioDim: t.audioDim, embeddingDim: c.captionChannels
        )
        _videoConnector.wrappedValue = LTXEmbeddingsConnector(
            dim: t.videoDim, heads: c.heads, headDim: c.videoHeadDim, layers: c.layers, registers: c.registers,
            ffMult: c.ffMult, maxPosition: c.maxPosition, doublePrecisionRope: t.doublePrecisionRope
        )
        _audioConnector.wrappedValue = LTXEmbeddingsConnector(
            dim: t.audioDim, heads: c.heads, headDim: c.audioHeadDim, layers: c.layers, registers: c.registers,
            ffMult: c.ffMult, maxPosition: c.maxPosition, doublePrecisionRope: t.doublePrecisionRope
        )
        super.init()
    }

    /// `GemmaFeaturesExtractorV2.__call__`: the hidden states stacked on a last axis, normalized
    /// per token over Gemma's width, flattened width-major, the padding zeroed; then projected and
    /// refined. Returns the video [1, T, 4096] and audio [1, T, 2048] contexts.
    func callAsFunction(hiddenStates: [MLXArray], attentionMask: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        let encoded = stacked(hiddenStates, axis: -1)
        let variance = (encoded * encoded).mean(axis: 2, keepDims: true)
        let normed = encoded * rsqrt(variance + 1e-6)
        let (batch, length, width, layers) = (normed.shape[0], normed.shape[1], normed.shape[2], normed.shape[3])
        var features = normed.reshaped([batch, length, width * layers])
        features = features * attentionMask.expandedDimensions(axis: 2).asType(features.dtype)
        eval(features)

        let (video, audio) = projection(features)
        let validTokens = attentionMask.sum().item(Int.self)
        return (videoConnector(video, validTokens: validTokens), audioConnector(audio, validTokens: validTokens))
    }

    /// The pack's keys for this module: `connector.` stripped, the list-wrapped layers renamed.
    static func weights(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for (key, value) in tensors {
            var name = key.hasPrefix("connector.") ? String(key.dropFirst("connector.".count)) : key
            name = name.replacingOccurrences(of: ".attn1.to_out.0.", with: ".attn1.to_out.")
                .replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
                .replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
            weights[name] = value
        }
        return weights
    }
}
