import Foundation
import MLX
import MLXNN

// LTX-2.5's DurationHead (`duration_head.safetensors`, dgrauet's `duration_head.py`): from the
// connector's video and audio contexts of a prompt, the length of the shot it describes, in
// seconds. Both streams are projected to a small width and tagged, one learned query attends over
// them, and an MLP gives the log of the duration. The dgrauet packs keep the pooler's q, k and v
// fused as torch's `in_proj_weight` / `in_proj_bias`.

final class LTXDurationCrossAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int

    init(hidden: Int, heads: Int) {
        self.heads = heads
        _qProj.wrappedValue = Linear(hidden, hidden)
        _kProj.wrappedValue = Linear(hidden, hidden)
        _vProj.wrappedValue = Linear(hidden, hidden)
        _outProj.wrappedValue = Linear(hidden, hidden)
        super.init()
    }

    func callAsFunction(_ queries: MLXArray, tokens: MLXArray) -> MLXArray {
        let (batch, count, length) = (queries.shape[0], queries.shape[1], tokens.shape[1])
        let headDim = queries.shape[2] / heads
        let q = qProj(queries).reshaped([batch, count, heads, headDim]).transposed(0, 2, 1, 3)
        let k = kProj(tokens).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(tokens).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped([batch, count, heads * headDim]))
    }
}

final class LTXDurationPooler: Module {
    @ParameterInfo(key: "query_tokens") var queryTokens: MLXArray
    @ModuleInfo(key: "cross_attn") var crossAttn: LTXDurationCrossAttention

    init(hidden: Int, queries: Int, heads: Int) {
        _queryTokens.wrappedValue = MLXArray.zeros([queries, hidden])
        _crossAttn.wrappedValue = LTXDurationCrossAttention(hidden: hidden, heads: heads)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let queries = broadcast(queryTokens.expandedDimensions(axis: 0),
                                to: [tokens.shape[0], queryTokens.shape[0], queryTokens.shape[1]])
        return crossAttn(queries, tokens: tokens)
    }
}

final class LTXDurationHead: Module {
    @ModuleInfo(key: "video_input_proj") var videoProj: Linear
    @ParameterInfo(key: "video_modality_emb") var videoEmbedding: MLXArray
    @ModuleInfo(key: "audio_input_proj") var audioProj: Linear
    @ParameterInfo(key: "audio_modality_emb") var audioEmbedding: MLXArray
    @ModuleInfo(key: "attention_pooler") var pooler: LTXDurationPooler
    @ModuleInfo(key: "mlp_hidden") var mlpHidden: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    /// The widths are read off the weights; the pooler's four heads are fixed upstream.
    init(videoDim: Int, audioDim: Int, hidden: Int, queries: Int, mlpDim: Int, heads: Int = 4) {
        _videoProj.wrappedValue = Linear(videoDim, hidden)
        _videoEmbedding.wrappedValue = MLXArray.zeros([hidden])
        _audioProj.wrappedValue = Linear(audioDim, hidden)
        _audioEmbedding.wrappedValue = MLXArray.zeros([hidden])
        _pooler.wrappedValue = LTXDurationPooler(hidden: hidden, queries: queries, heads: heads)
        _mlpHidden.wrappedValue = Linear(hidden * queries, mlpDim)
        _mlpOut.wrappedValue = Linear(mlpDim, 1)
        super.init()
    }

    /// Seconds [B] from the connector's video [B, T, 4096] and audio [B, T, 2048] contexts.
    func callAsFunction(video: MLXArray, audio: MLXArray) -> MLXArray {
        let tokens = concatenated([videoProj(video) + videoEmbedding, audioProj(audio) + audioEmbedding], axis: 1)
        let pooled = pooler(tokens)
        let hidden = geluApproximate(mlpHidden(pooled.reshaped([pooled.shape[0], -1])))
        return exp(mlpOut(hidden).squeezed(axis: -1))
    }

    /// Loads `duration_head.safetensors`: the prefix stripped, the fused q/k/v split.
    static func load(_ url: URL) throws -> LTXDurationHead {
        var weights = stripping("duration_head.", from: try loadArrays(url: url))
        let fused = "attention_pooler.cross_attn.in_proj_"
        if let weight = weights.removeValue(forKey: fused + "weight"), let bias = weights.removeValue(forKey: fused + "bias") {
            let (ws, bs) = (split(weight, parts: 3, axis: 0), split(bias, parts: 3, axis: 0))
            for (index, name) in ["q_proj", "k_proj", "v_proj"].enumerated() {
                weights["attention_pooler.cross_attn.\(name).weight"] = ws[index]
                weights["attention_pooler.cross_attn.\(name).bias"] = bs[index]
            }
        }
        guard let embedding = weights["video_modality_emb"], let video = weights["video_input_proj.weight"],
              let audio = weights["audio_input_proj.weight"], let mlp = weights["mlp_hidden.weight"],
              let queries = weights["attention_pooler.query_tokens"]
        else { throw LTXError.missingFile("duration head weights", url.deletingLastPathComponent()) }
        let head = LTXDurationHead(videoDim: video.shape[1], audioDim: audio.shape[1], hidden: embedding.shape[0],
                                   queries: queries.shape[0], mlpDim: mlp.shape[0])
        try WeightLoading.apply(weights, to: head)
        return head
    }

    /// `seconds_to_clamped_num_frames`: the seconds at `fps` rounded (half to even), clamped to
    /// [minFrames, maxFrames], floored to the 8k + 1 grid, or raised to the next grid point when
    /// that falls under the minimum.
    static func frames(seconds: Double, fps: Double, minFrames: Int, maxFrames: Int) -> Int {
        let raw = Int((seconds * fps).rounded(.toNearestOrEven))
        let clamped = max(minFrames, min(raw, maxFrames))
        let t = LTXConfig.temporalScale
        var frames = ((clamped - 1) / t) * t + 1
        if frames < minFrames {
            frames = min(((minFrames - 1 + t - 1) / t) * t + 1, maxFrames)
        }
        return frames
    }
}
