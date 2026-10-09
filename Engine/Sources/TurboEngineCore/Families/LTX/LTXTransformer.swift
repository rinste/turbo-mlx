import Foundation
import MLX
import MLXNN

// LTX-2.3's diffusion transformer: 48 blocks that carry a video stream (4096 wide) and an audio
// stream (2048 wide) side by side, each with self-attention, text cross-attention and a feed
// forward, and the two streams attending to each other in the middle of every block. After
// dgrauet's `model/transformer/` (`LTXModel`, `BasicAVTransformerBlock`, `Attention`), module
// for module, so the pack's `transformer.*` keys load unchanged; the block linears are stored
// quantized (4 or 8 bits), everything else in bfloat16.
//
// The modulation parameters come out of their MLPs in float32 (the sinusoidal timestep embedding
// is float32), and the reference lets them promote the streams: from the first block on, the
// activations are float32 while the weights stay bfloat16 or quantized. The port keeps that.

final class LTXAttention: Module {
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

    init(queryDim: Int, kvDim: Int? = nil, outDim: Int? = nil, heads: Int, headDim: Int, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        scale = 1 / Float(headDim).squareRoot()
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(queryDim, inner, bias: true)
        _toK.wrappedValue = Linear(kvDim ?? queryDim, inner, bias: true)
        _toV.wrappedValue = Linear(kvDim ?? queryDim, inner, bias: true)
        _toOut.wrappedValue = Linear(inner, outDim ?? queryDim, bias: true)
        _toGateLogits.wrappedValue = Linear(queryDim, heads, bias: true)
        _qNorm.wrappedValue = RMSNorm(dimensions: inner, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: inner, eps: eps)
        super.init()
    }

    /// Self-attention without `context`, cross-attention with it. `rope` rotates the queries (and
    /// the keys, unless `keyRope` is given); `mask` is additive. `valuesOnly`: STG's perturbation,
    /// the attention replaced by the values (the reference's `out · 0 + v · 1`, the same numbers).
    func callAsFunction(
        _ x: MLXArray, context: MLXArray? = nil, rope: LTXRope? = nil, keyRope: LTXRope? = nil, mask: MLXArray? = nil,
        valuesOnly: Bool = false
    ) -> MLXArray {
        let batch = x.shape[0]
        let source = context ?? x
        var q = qNorm(toQ(x)).reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        var k = kNorm(toK(source)).reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        let v = toV(source).reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        var out: MLXArray
        if valuesOnly {
            out = v
        } else {
            if let rope {
                q = rope.apply(q)
                k = (keyRope ?? rope).apply(k)
            }
            out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        }
        let gate = 2 * sigmoid(toGateLogits(x))
        out = out * gate.transposed(0, 2, 1).expandedDimensions(axis: -1)
        return toOut(out.transposed(0, 2, 1, 3).reshaped([batch, -1, heads * headDim]))
    }
}

final class LTXFeedForward: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dim: Int, mult: Int, bias: Bool) {
        _projIn.wrappedValue = Linear(dim, dim * mult, bias: bias)
        _projOut.wrappedValue = Linear(dim * mult, dim, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projOut(geluApproximate(projIn(x)))
    }
}

final class LTXTimestepEmbedding: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(inChannels: Int, dim: Int) {
        _linear1.wrappedValue = Linear(inChannels, dim)
        _linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

final class LTXTimestepEmbedder: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: LTXTimestepEmbedding

    init(inChannels: Int, dim: Int) {
        _timestepEmbedder.wrappedValue = LTXTimestepEmbedding(inChannels: inChannels, dim: dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { timestepEmbedder(x) }
}

/// `AdaLayerNormSingle`: from a timestep embedding to `count` modulation vectors of `dim`, and
/// the intermediate embedding the output block uses.
final class LTXAdaLayerNormSingle: Module {
    @ModuleInfo(key: "emb") var emb: LTXTimestepEmbedder
    @ModuleInfo(key: "linear") var linear: Linear

    let count: Int

    init(dim: Int, count: Int, timestepDim: Int) {
        self.count = count
        _emb.wrappedValue = LTXTimestepEmbedder(inChannels: timestepDim, dim: dim)
        _linear.wrappedValue = Linear(dim, count * dim)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> (params: MLXArray, embedded: MLXArray) {
        let embedded = emb(timestep)
        return (linear(silu(embedded)), embedded)
    }
}

/// Modulation parameters: one row per sample, or one per token given as the distinct rows and the
/// row each token takes (few distinct timesteps: the conditioned first frame and the rest).
enum LTXModulation {
    case shared(MLXArray)
    case perToken(rows: MLXArray, index: MLXArray, tokens: Int)

    /// `_unpack_adaln`: the `count` vectors after adding the block's table, each broadcastable to
    /// [B, N, dim].
    func unpack(table: MLXArray, count: Int, dim: Int) -> [MLXArray] {
        switch self {
        case .shared(let params):
            let p = params.reshaped([-1, count, dim]) + table[.newAxis, 0 ..< count, 0...]
            return (0 ..< count).map { p[0..., $0, 0...].expandedDimensions(axis: 1) }
        case .perToken(let rows, let index, let tokens):
            let total = rows.shape[1] / dim
            let p = rows.reshaped([-1, total, dim])[0..., 0 ..< count, 0...] + table[.newAxis, 0 ..< count, 0...]
            return (0 ..< count).map { p[0..., $0, 0...].take(index, axis: 0).reshaped([1, tokens, dim]) }
        }
    }

    /// The whole [B, N or 1, dim] tensor (the output block's embedded timestep).
    func gathered() -> MLXArray {
        switch self {
        case .shared(let params): return params.expandedDimensions(axis: 1)
        case .perToken(let rows, let index, let tokens): return rows.take(index, axis: 0).reshaped([1, tokens, -1])
        }
    }
}

/// `get_timestep_embedding` (flip to [cos, sin], no shift): [N] → [N, dim] float32.
func ltxTimestepEmbedding(_ timesteps: MLXArray, dim: Int) -> MLXArray {
    let half = dim / 2
    let exponent = MLXArray(Float(-log(10000.0))) * MLXArray(0 ..< half).asType(.float32) / Float(half)
    let frequencies = exp(exponent)
    let args = timesteps.expandedDimensions(axis: 1).asType(.float32) * frequencies.expandedDimensions(axis: 0)
    return concatenated([cos(args), sin(args)], axis: -1)
}

final class LTXTransformerBlock: Module {
    @ModuleInfo(key: "attn1") var attn1: LTXAttention
    @ModuleInfo(key: "audio_attn1") var audioAttn1: LTXAttention
    @ModuleInfo(key: "attn2") var attn2: LTXAttention
    @ModuleInfo(key: "audio_attn2") var audioAttn2: LTXAttention
    @ModuleInfo(key: "audio_to_video_attn") var audioToVideoAttn: LTXAttention
    @ModuleInfo(key: "video_to_audio_attn") var videoToAudioAttn: LTXAttention
    @ModuleInfo(key: "ff") var ff: LTXFeedForward
    @ModuleInfo(key: "audio_ff") var audioFF: LTXFeedForward
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ParameterInfo(key: "prompt_scale_shift_table") var promptScaleShiftTable: MLXArray
    @ParameterInfo(key: "audio_prompt_scale_shift_table") var audioPromptScaleShiftTable: MLXArray
    @ParameterInfo(key: "scale_shift_table_a2v_ca_video") var crossVideoTable: MLXArray
    @ParameterInfo(key: "scale_shift_table_a2v_ca_audio") var crossAudioTable: MLXArray

    let eps: Float

    init(config c: LTXConfig.Transformer) {
        eps = c.normEps
        _attn1.wrappedValue = LTXAttention(queryDim: c.videoDim, heads: c.videoHeads, headDim: c.videoHeadDim, eps: c.normEps)
        _audioAttn1.wrappedValue = LTXAttention(queryDim: c.audioDim, heads: c.audioHeads, headDim: c.audioHeadDim, eps: c.normEps)
        _attn2.wrappedValue = LTXAttention(queryDim: c.videoDim, heads: c.videoHeads, headDim: c.videoHeadDim, eps: c.normEps)
        _audioAttn2.wrappedValue = LTXAttention(queryDim: c.audioDim, heads: c.audioHeads, headDim: c.audioHeadDim, eps: c.normEps)
        _audioToVideoAttn.wrappedValue = LTXAttention(
            queryDim: c.videoDim, kvDim: c.audioDim, outDim: c.videoDim, heads: c.crossHeads, headDim: c.crossHeadDim, eps: c.normEps
        )
        _videoToAudioAttn.wrappedValue = LTXAttention(
            queryDim: c.audioDim, kvDim: c.videoDim, outDim: c.audioDim, heads: c.crossHeads, headDim: c.crossHeadDim, eps: c.normEps
        )
        _ff.wrappedValue = LTXFeedForward(dim: c.videoDim, mult: c.ffMult, bias: c.ffBias)
        _audioFF.wrappedValue = LTXFeedForward(dim: c.audioDim, mult: c.ffMult, bias: c.audioFFBias)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([9, c.videoDim])
        _audioScaleShiftTable.wrappedValue = MLXArray.zeros([9, c.audioDim])
        _promptScaleShiftTable.wrappedValue = MLXArray.zeros([2, c.videoDim])
        _audioPromptScaleShiftTable.wrappedValue = MLXArray.zeros([2, c.audioDim])
        _crossVideoTable.wrappedValue = MLXArray.zeros([5, c.videoDim])
        _crossAudioTable.wrappedValue = MLXArray.zeros([5, c.audioDim])
        super.init()
    }

    private func norm(_ x: MLXArray) -> MLXArray { unweightedRMSNorm(x, eps: eps) }

    /// One block over both streams (`BasicAVTransformerBlock.__call__`, without attention masks,
    /// which neither pipeline uses), with the guidance's perturbations when `context` has them.
    func callAsFunction(video: MLXArray, audio: MLXArray, context: LTXBlockContext, index: Int) -> (video: MLXArray, audio: MLXArray) {
        let perturbation = context.perturbation
        let vdim = video.shape[2]
        let adim = audio.shape[2]
        var video = video
        var audio = audio

        // Nine vectors per stream: [0-2] self-attention, [3-5] feed forward, [6-8] text.
        let v = context.video.unpack(table: scaleShiftTable, count: 9, dim: vdim)
        let a = context.audio.unpack(table: audioScaleShiftTable, count: 9, dim: adim)
        // Cross-modal: [0] scale a→v, [1] shift a→v, [2] scale v→a, [3] shift v→a; [4] the gate.
        let cv = context.crossVideo.unpack(table: crossVideoTable, count: 4, dim: vdim)
        let ca = context.crossAudio.unpack(table: crossAudioTable, count: 4, dim: adim)
        let gateToVideo = (context.crossGateToVideo + crossVideoTable[4, 0...]).expandedDimensions(axis: 1)
        let gateToAudio = (context.crossGateToAudio + crossAudioTable[4, 0...]).expandedDimensions(axis: 1)

        // 1–2. Self-attention.
        video = video + attn1(norm(video) * (1 + v[1]) + v[0], rope: context.videoRope,
                              valuesOnly: perturbation.skipsVideoSelfAttention.contains(index)) * v[2]
        audio = audio + audioAttn1(norm(audio) * (1 + a[1]) + a[0], rope: context.audioRope,
                                   valuesOnly: perturbation.skipsAudioSelfAttention.contains(index)) * a[2]

        // 3–4. Text cross-attention, the text modulated by the prompt tables.
        let vp = LTXModulation.shared(context.videoPrompt).unpack(table: promptScaleShiftTable, count: 2, dim: vdim)
        let videoText = context.videoText * (1 + vp[1]) + vp[0]
        video = video + attn2(norm(video) * (1 + v[7]) + v[6], context: videoText) * v[8]
        let ap = LTXModulation.shared(context.audioPrompt).unpack(table: audioPromptScaleShiftTable, count: 2, dim: adim)
        let audioText = context.audioText * (1 + ap[1]) + ap[0]
        audio = audio + audioAttn2(norm(audio) * (1 + a[7]) + a[6], context: audioText) * a[8]

        // 5–6. Audio ↔ video, both from the same normalized streams; left out when the guidance
        // isolates the modalities (the reference multiplies them by zero).
        if !perturbation.isolatesModalities {
            let videoNorm = norm(video)
            let audioNorm = norm(audio)
            let toVideo = audioToVideoAttn(
                videoNorm * (1 + cv[0]) + cv[1], context: audioNorm * (1 + ca[0]) + ca[1],
                rope: context.videoCrossRope, keyRope: context.audioCrossRope
            ) * gateToVideo
            video = video + toVideo
            let toAudio = videoToAudioAttn(
                audioNorm * (1 + ca[2]) + ca[3], context: videoNorm * (1 + cv[2]) + cv[3],
                rope: context.audioCrossRope, keyRope: context.videoCrossRope
            ) * gateToAudio
            audio = audio + toAudio
        }

        // 7–8. Feed forward.
        video = video + ff(norm(video) * (1 + v[4]) + v[3]) * v[5]
        audio = audio + audioFF(norm(audio) * (1 + a[4]) + a[3]) * a[5]
        return (video, audio)
    }
}

/// What a guided pass changes in the blocks (`PerturbationConfig`): STG's self-attentions turned
/// into their values in some blocks, and the modality guidance's pass with the audio–video
/// cross-attentions left out of every block.
public struct LTXPerturbation {
    var skipsVideoSelfAttention: Set<Int> = []
    var skipsAudioSelfAttention: Set<Int> = []
    var isolatesModalities = false

    public static let none = LTXPerturbation()
}

/// What every block of one forward pass shares.
struct LTXBlockContext {
    var perturbation = LTXPerturbation.none
    var video: LTXModulation
    var audio: LTXModulation
    var crossVideo: LTXModulation
    var crossAudio: LTXModulation
    /// [B, dim]: the gates are always per sample.
    var crossGateToVideo: MLXArray
    var crossGateToAudio: MLXArray
    var videoPrompt: MLXArray
    var audioPrompt: MLXArray
    var videoText: MLXArray
    var audioText: MLXArray
    var videoRope: LTXRope
    var audioRope: LTXRope
    var videoCrossRope: LTXRope
    var audioCrossRope: LTXRope
}

final class LTXTransformer: Module {
    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "audio_patchify_proj") var audioPatchifyProj: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray
    @ModuleInfo(key: "adaln_single") var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_adaln_single") var audioAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "prompt_adaln_single") var promptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_prompt_adaln_single") var audioPromptAdalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") var crossVideoAdaln: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") var crossAudioAdaln: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") var gateToVideoAdaln: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") var gateToAudioAdaln: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") var blocks: [LTXTransformerBlock]
    /// LTX-2.5: the learned marker added to the tokens of single-pixel latent frames.
    @ParameterInfo(key: "keyframes_abs_pos_embedding") var keyframesEmbedding: MLXArray?

    let config: LTXConfig.Transformer
    /// Blocks run between two evaluations (`LTX2_DIT_EVAL_EVERY`): short Metal command buffers.
    var evalEvery = 8

    init(config c: LTXConfig.Transformer) {
        config = c
        let t = c.timestepEmbeddingDim
        _patchifyProj.wrappedValue = Linear(c.videoPatchChannels, c.videoDim)
        _audioPatchifyProj.wrappedValue = Linear(c.audioPatchChannels, c.audioDim)
        _projOut.wrappedValue = Linear(c.videoDim, c.videoPatchChannels)
        _audioProjOut.wrappedValue = Linear(c.audioDim, c.audioPatchChannels)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([2, c.videoDim])
        _audioScaleShiftTable.wrappedValue = MLXArray.zeros([2, c.audioDim])
        _adalnSingle.wrappedValue = LTXAdaLayerNormSingle(dim: c.videoDim, count: 9, timestepDim: t)
        _audioAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(dim: c.audioDim, count: 9, timestepDim: t)
        _promptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(dim: c.videoDim, count: 2, timestepDim: t)
        _audioPromptAdalnSingle.wrappedValue = LTXAdaLayerNormSingle(dim: c.audioDim, count: 2, timestepDim: t)
        _crossVideoAdaln.wrappedValue = LTXAdaLayerNormSingle(dim: c.videoDim, count: 4, timestepDim: t)
        _crossAudioAdaln.wrappedValue = LTXAdaLayerNormSingle(dim: c.audioDim, count: 4, timestepDim: t)
        _gateToVideoAdaln.wrappedValue = LTXAdaLayerNormSingle(dim: c.videoDim, count: 1, timestepDim: t)
        _gateToAudioAdaln.wrappedValue = LTXAdaLayerNormSingle(dim: c.audioDim, count: 1, timestepDim: t)
        _blocks.wrappedValue = (0 ..< c.numLayers).map { _ in LTXTransformerBlock(config: c) }
        _keyframesEmbedding.wrappedValue = c.keyframesEmbedding ? MLXArray.zeros([1, c.videoDim]) : nil
        super.init()
    }

    /// Per-token modulation from per-token timesteps [B, N] that take a few distinct values: the
    /// MLP runs on the distinct rows only.
    private func perToken(_ module: LTXAdaLayerNormSingle, timesteps: MLXArray) -> (params: LTXModulation, embedded: LTXModulation) {
        let flat = timesteps.reshaped([-1])
        let values = flat.asType(.float32).asArray(Float.self)
        var distinct: [Float] = []
        var index: [Int32] = []
        for value in values {
            if let position = distinct.firstIndex(of: value) {
                index.append(Int32(position))
            } else {
                distinct.append(value)
                index.append(Int32(distinct.count - 1))
            }
        }
        let rows = MLXArray(distinct).asType(timesteps.dtype) * config.timestepScale
        let (params, embedded) = module(ltxTimestepEmbedding(rows, dim: config.timestepEmbeddingDim))
        let tokens = flat.shape[0]
        let indices = MLXArray(index)
        return (.perToken(rows: params, index: indices, tokens: tokens), .perToken(rows: embedded, index: indices, tokens: tokens))
    }

    /// One forward pass (`LTXModel.__call__`): the velocities of both streams.
    ///
    /// - Parameters:
    ///   - video: [B, Nv, 128] patchified video latent; `audio` [B, Na, 128].
    ///   - sigma: [B] bfloat16, the step's noise level.
    ///   - videoTimesteps: [B, Nv] per-token timesteps when some tokens are conditioning (nil: all
    ///     at `sigma`).
    ///   - keyframeTokens: the first tokens, which get the keyframe marker (2.5: the first latent
    ///     frame, one pixel frame); ignored by a model without it.
    func callAsFunction(
        video: MLXArray, audio: MLXArray, sigma: MLXArray, videoTimesteps: MLXArray?,
        videoText: MLXArray, audioText: MLXArray, videoPositions: MLXArray, audioPositions: MLXArray,
        keyframeTokens: Int = 0, perturbation: LTXPerturbation = .none
    ) -> (video: MLXArray, audio: MLXArray) {
        let c = config
        var videoHidden = patchifyProj(video.asType(.bfloat16))
        // `apply_keyframes_absolute_embedding`: the marker times a 0/1 mask, added to every token.
        if let keyframesEmbedding, keyframeTokens > 0 {
            let count = videoHidden.shape[1]
            let marked = min(keyframeTokens, count)
            let mask = concatenated([MLXArray.ones([videoHidden.shape[0], marked, 1], dtype: videoHidden.dtype),
                                     MLXArray.zeros([videoHidden.shape[0], count - marked, 1], dtype: videoHidden.dtype)], axis: 1)
            videoHidden = videoHidden + mask * keyframesEmbedding.asType(videoHidden.dtype)
        }
        var audioHidden = audioPatchifyProj(audio.asType(.bfloat16))

        let timestep = sigma.asType(.bfloat16)
        let timestepEmbedding = ltxTimestepEmbedding(timestep * c.timestepScale, dim: c.timestepEmbeddingDim)
        let gateFactor = c.crossTimestepScale / c.timestepScale
        let gateEmbedding = ltxTimestepEmbedding(timestep * c.timestepScale * gateFactor, dim: c.timestepEmbeddingDim)

        let videoModulation: LTXModulation
        let videoEmbedded: LTXModulation
        let crossVideo: LTXModulation
        if let videoTimesteps {
            (videoModulation, videoEmbedded) = perToken(adalnSingle, timesteps: videoTimesteps)
            crossVideo = perToken(crossVideoAdaln, timesteps: videoTimesteps).params
        } else {
            let (params, embedded) = adalnSingle(timestepEmbedding)
            (videoModulation, videoEmbedded) = (.shared(params), .shared(embedded))
            crossVideo = .shared(crossVideoAdaln(timestepEmbedding).params)
        }
        let (audioParams, audioEmbeddedTimestep) = audioAdalnSingle(timestepEmbedding)

        let context = LTXBlockContext(
            perturbation: perturbation,
            video: videoModulation,
            audio: .shared(audioParams),
            crossVideo: crossVideo,
            crossAudio: .shared(crossAudioAdaln(timestepEmbedding).params),
            crossGateToVideo: gateToVideoAdaln(gateEmbedding).params,
            crossGateToAudio: gateToAudioAdaln(gateEmbedding).params,
            videoPrompt: promptAdalnSingle(timestepEmbedding).params,
            audioPrompt: audioPromptAdalnSingle(timestepEmbedding).params,
            videoText: videoText.asType(.bfloat16),
            audioText: audioText.asType(.bfloat16),
            videoRope: LTXRope(positions: videoPositions, innerDim: c.videoHeads * c.videoHeadDim, heads: c.videoHeads,
                               theta: c.ropeTheta, maxPositions: c.maxPositions, doublePrecision: c.doublePrecisionRope),
            audioRope: LTXRope(positions: audioPositions, innerDim: c.audioHeads * c.audioHeadDim, heads: c.audioHeads,
                               theta: c.ropeTheta, maxPositions: c.audioMaxPositions, doublePrecision: c.doublePrecisionRope),
            videoCrossRope: LTXRope(positions: videoPositions[0..., 0..., 0 ..< 1], innerDim: c.crossHeads * c.crossHeadDim,
                                    heads: c.crossHeads, theta: c.ropeTheta,
                                    maxPositions: [max(c.maxPositions[0], c.audioMaxPositions[0])], doublePrecision: c.doublePrecisionRope),
            audioCrossRope: LTXRope(positions: audioPositions[0..., 0..., 0 ..< 1], innerDim: c.crossHeads * c.crossHeadDim,
                                    heads: c.crossHeads, theta: c.ropeTheta,
                                    maxPositions: [max(c.maxPositions[0], c.audioMaxPositions[0])], doublePrecision: c.doublePrecisionRope)
        )

        for (index, block) in blocks.enumerated() {
            (videoHidden, audioHidden) = block(video: videoHidden, audio: audioHidden, context: context, index: index)
            if evalEvery > 0, (index + 1) % evalEvery == 0 { eval(videoHidden, audioHidden) }
        }

        let videoOut = outputBlock(videoHidden, embedded: videoEmbedded.gathered(), table: scaleShiftTable, projection: projOut)
        let audioOut = outputBlock(audioHidden, embedded: audioEmbeddedTimestep.expandedDimensions(axis: 1),
                                   table: audioScaleShiftTable, projection: audioProjOut)
        return (videoOut, audioOut)
    }

    /// Layer norm without affine, the table plus the embedded timestep as shift and scale, then the
    /// projection back to latent channels.
    private func outputBlock(_ x: MLXArray, embedded: MLXArray, table: MLXArray, projection: Linear) -> MLXArray {
        let values = table.expandedDimensions(axes: [0, 1]) + embedded.expandedDimensions(axis: 2)
        let shift = values[0..., 0..., 0, 0...]
        let scale = values[0..., 0..., 1, 0...]
        let normed = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: config.normEps)
        return projection(normed * (1 + scale) + shift)
    }

    /// The pack's keys: `transformer.` stripped.
    static func weights(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for (key, value) in tensors where key.hasPrefix("transformer.") {
            weights[String(key.dropFirst("transformer.".count))] = value
        }
        return weights
    }
}
