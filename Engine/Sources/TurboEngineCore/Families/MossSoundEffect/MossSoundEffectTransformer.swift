import Foundation
import MLX
import MLXNN

// MOSS-SoundEffect v2.0's DiT, after `WanAudioModel` and `DiTBlock` in OpenMOSS's MOSS-TTS
// (`moss_soundeffect_v2/diffsynth/models/wan_audio_dit.py`, `wan_video_dit.py`) as `model_fn_wan_video` runs it: Wan
// 2.1's 1.3B block over the 128-channel latents of a 30 s clip, one token per latent frame (a 1-wide "patch"), with a
// plain 1-D rotary embedding over each 128-wide head. The dtype points are those of the pipeline's bf16 autocast: the
// timestep embedding, every layer norm, the modulation and the residual stream in float32, the projections and the
// attention in the weights' dtype, RoPE in float32 on interleaved pairs. Tokens are channels-last ([B, T, C]); the
// model takes and returns latents channels-second ([B, 128, T]) as the reference does.

/// `torch.nn.RMSNorm` over the whole width (q and k before the heads split), in float32.
final class MossSFXRMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
    }
}

/// The affine `nn.LayerNorm` (`norm3`), which autocast runs in float32.
final class MossSFXLayerNorm: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        _bias.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x.asType(.float32), weight: weight.asType(.float32), bias: bias.asType(.float32), eps: eps)
    }
}

/// `nn.LayerNorm(elementwise_affine=False)` in float32.
func mossSFXLayerNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.layerNorm(x.asType(.float32), weight: nil, bias: nil, eps: eps)
}

/// `GELU(approximate='tanh')` as torch computes it on 16-bit tensors: in float32, rounded once.
func mossSFXGELU(_ x: MLXArray) -> MLXArray {
    geluApproximate(x.asType(.float32)).asType(x.dtype)
}

/// The rotary phases of `precompute_freqs_cis_1d(128)` for positions 0..<frames: cos and sin [frames, 1, 64] in
/// float32, worked out in float64 as the reference does (its three chunks put back together are one plain table).
public struct MossSFXRotary {
    public let cos: MLXArray
    public let sin: MLXArray

    public init(frames: Int, headDim: Int, theta: Double = 10_000) {
        let half = headDim / 2
        var cosines = [Float](repeating: 0, count: frames * half)
        var sines = [Float](repeating: 0, count: frames * half)
        for position in 0 ..< frames {
            for index in 0 ..< half {
                let angle = Double(position) / pow(theta, Double(2 * index) / Double(headDim))
                cosines[position * half + index] = Float(Foundation.cos(angle))
                sines[position * half + index] = Float(Foundation.sin(angle))
            }
        }
        cos = MLXArray(cosines, [frames, 1, half])
        sin = MLXArray(sines, [frames, 1, half])
    }

    /// `rope_apply`: [B, T, H·D] in float32, each head's (even, odd) pairs rotated; back in `dtype`.
    func apply(_ x: MLXArray, heads: Int, dtype: DType) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let pairs = x.asType(.float32).reshaped([batch, length, heads, width / heads / 2, 2])
        let even = pairs[.ellipsis, 0]
        let odd = pairs[.ellipsis, 1]
        let c = cos[0 ..< length], s = sin[0 ..< length]
        let rotated = stacked([even * c - odd * s, even * s + odd * c], axis: -1)
        return rotated.reshaped([batch, length, width]).asType(dtype)
    }
}

/// `SelfAttention` and `CrossAttention` (no image branch): biased projections, q and k RMS-normalized over the
/// whole width, rotary phases on the self-attention only, attention unmasked.
final class MossSFXAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: MossSFXRMSNorm
    @ModuleInfo(key: "norm_k") var normK: MossSFXRMSNorm

    let heads: Int

    init(dim: Int, heads: Int, eps: Float) {
        self.heads = heads
        _q.wrappedValue = Linear(dim, dim)
        _k.wrappedValue = Linear(dim, dim)
        _v.wrappedValue = Linear(dim, dim)
        _o.wrappedValue = Linear(dim, dim)
        _normQ.wrappedValue = MossSFXRMSNorm(dim, eps: eps)
        _normK.wrappedValue = MossSFXRMSNorm(dim, eps: eps)
        super.init()
    }

    /// `x` [B, T, dim] (float32) attends to `context` [B, S, dim] (in `dtype`), or to itself with `rotary`.
    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil, rotary: MossSFXRotary? = nil, dtype: DType) -> MLXArray {
        let input = x.asType(dtype)
        let source = context ?? input
        var queries = normQ(q(input))
        var keys = normK(k(source))
        if let rotary {
            queries = rotary.apply(queries, heads: heads, dtype: dtype)
            keys = rotary.apply(keys, heads: heads, dtype: dtype)
        }
        let values = v(source)
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = width / heads
        func split(_ t: MLXArray) -> MLXArray {
            t.asType(dtype).reshaped([batch, t.dim(1), heads, headDim]).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(queries), keys: split(keys), values: split(values),
            scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return o(attended.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

/// `DiTBlock`: modulated self-attention, plain cross-attention on the affine `norm3`, modulated feed-forward.
final class MossSFXBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MossSFXAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: MossSFXAttention
    @ModuleInfo(key: "norm3") var norm3: MossSFXLayerNorm
    @ModuleInfo(key: "ffn_in") var ffnIn: Linear
    @ModuleInfo(key: "ffn_out") var ffnOut: Linear
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    let eps: Float

    init(config: MossSoundEffectConfig) {
        eps = config.eps
        _selfAttn.wrappedValue = MossSFXAttention(dim: config.dim, heads: config.heads, eps: config.eps)
        _crossAttn.wrappedValue = MossSFXAttention(dim: config.dim, heads: config.heads, eps: config.eps)
        _norm3.wrappedValue = MossSFXLayerNorm(config.dim, eps: config.eps)
        _ffnIn.wrappedValue = Linear(config.dim, config.ffnDim)
        _ffnOut.wrappedValue = Linear(config.ffnDim, config.dim)
        _modulation.wrappedValue = MLXArray.zeros([1, 6, config.dim])
        super.init()
    }

    /// `x` [B, T, dim] float32, `timeModulation` [B, 6, dim] float32, `context` [B, S, dim] in `dtype`.
    func callAsFunction(_ x: MLXArray, context: MLXArray, timeModulation: MLXArray, rotary: MossSFXRotary, dtype: DType) -> MLXArray {
        let table = modulation.asType(.float32) + timeModulation
        func part(_ index: Int) -> MLXArray { table[0..., index ..< index + 1] }
        let attended = selfAttn(mossSFXLayerNorm(x, eps: eps) * (1 + part(1)) + part(0), rotary: rotary, dtype: dtype)
        var h = x + part(2) * attended.asType(.float32)
        h = h + crossAttn(norm3(h), context: context, dtype: dtype).asType(.float32)
        let y = (mossSFXLayerNorm(h, eps: eps) * (1 + part(4)) + part(3)).asType(dtype)
        return h + part(5) * ffnOut(mossSFXGELU(ffnIn(y))).asType(.float32)
    }
}

/// `Head`: the last layer norm modulated by the timestep embedding (not its projection), then the output linear.
final class MossSFXHead: Module {
    @ModuleInfo(key: "head") var head: Linear
    @ParameterInfo(key: "modulation") var modulation: MLXArray

    let eps: Float

    init(config: MossSoundEffectConfig) {
        eps = config.eps
        _head.wrappedValue = Linear(config.dim, config.latentChannels)
        _modulation.wrappedValue = MLXArray.zeros([1, 2, config.dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, time: MLXArray, dtype: DType) -> MLXArray {
        let table = modulation.asType(.float32) + time.expandedDimensions(axis: 1)
        let shift = table[0..., 0 ..< 1], scale = table[0..., 1 ..< 2]
        return head((mossSFXLayerNorm(x, eps: eps) * (1 + scale) + shift).asType(dtype))
    }
}

public final class MossSoundEffectTransformer: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "text_embedding_in") var textEmbeddingIn: Linear
    @ModuleInfo(key: "text_embedding_out") var textEmbeddingOut: Linear
    @ModuleInfo(key: "time_embedding_in") var timeEmbeddingIn: Linear
    @ModuleInfo(key: "time_embedding_out") var timeEmbeddingOut: Linear
    @ModuleInfo(key: "time_projection") var timeProjection: Linear
    @ModuleInfo(key: "blocks") var blocks: [MossSFXBlock]
    @ModuleInfo(key: "head") var head: MossSFXHead

    let config: MossSoundEffectConfig
    /// The rotary phases of the longest clip; shorter inputs read their first rows.
    let rotary: MossSFXRotary

    public init(config: MossSoundEffectConfig) {
        self.config = config
        _patchEmbedding.wrappedValue = Linear(config.latentChannels, config.dim)
        _textEmbeddingIn.wrappedValue = Linear(config.textDim, config.dim)
        _textEmbeddingOut.wrappedValue = Linear(config.dim, config.dim)
        _timeEmbeddingIn.wrappedValue = Linear(config.freqDim, config.dim)
        _timeEmbeddingOut.wrappedValue = Linear(config.dim, config.dim)
        _timeProjection.wrappedValue = Linear(config.dim, 6 * config.dim)
        _blocks.wrappedValue = (0 ..< config.layers).map { _ in MossSFXBlock(config: config) }
        _head.wrappedValue = MossSFXHead(config: config)
        rotary = MossSFXRotary(frames: config.latentFrames, headDim: config.dim / config.heads)
        super.init()
    }

    /// The dtype the projections run in: the weights' (bf16 in the MLX pack, float32 in OpenMOSS's checkpoint).
    public var computeType: DType { patchEmbedding.weight.dtype == .float32 ? .float32 : .bfloat16 }

    /// `text_embedding` on the encoder's states [B, 512, 2048]: what every block's cross-attention reads, in the
    /// compute dtype. The same at every step, so the pipeline works it out once a prompt.
    public func embedText(_ states: MLXArray) -> MLXArray {
        textEmbeddingOut(mossSFXGELU(textEmbeddingIn(states.asType(computeType))))
    }

    /// `sinusoidal_embedding_1d(256, t)`: [cos | sin] of t·10000^(−i/128), in float64, rounded to float32.
    static func sinusoid(_ timestep: Double, dim: Int) -> MLXArray {
        let half = dim / 2
        var values = [Float](repeating: 0, count: dim)
        for index in 0 ..< half {
            let angle = timestep * pow(10_000, -Double(index) / Double(half))
            values[index] = Float(Foundation.cos(angle))
            values[half + index] = Float(Foundation.sin(angle))
        }
        return MLXArray(values, [1, dim])
    }

    /// The timestep embedding [1, dim] and its six-fold projection [1, 6, dim], in float32.
    public func timeEmbedding(_ timestep: Double) -> (time: MLXArray, modulation: MLXArray) {
        func float32(_ linear: Linear, _ x: MLXArray) -> MLXArray {
            let y = matmul(x, linear.weight.asType(.float32).T)
            return linear.bias.map { y + $0.asType(.float32) } ?? y
        }
        let time = float32(timeEmbeddingOut, silu(float32(timeEmbeddingIn, Self.sinusoid(timestep, dim: config.freqDim))))
        let modulation = float32(timeProjection, silu(time)).reshaped([1, 6, config.dim])
        return (time, modulation)
    }

    /// The velocity [B, 128, T] in the compute dtype for latents [B, 128, T] at a timestep (0…1000), the prompt
    /// already through `embedText`.
    public func callAsFunction(_ latents: MLXArray, timestep: Double, text: MLXArray) -> MLXArray {
        let (time, modulation) = timeEmbedding(timestep)
        return callAsFunction(latents, time: time, modulation: modulation, text: text)
    }

    /// The same with the timestep's embedding worked out (once a step for both of guidance's passes).
    public func callAsFunction(_ latents: MLXArray, time: MLXArray, modulation: MLXArray, text: MLXArray) -> MLXArray {
        let dtype = computeType
        var x = patchEmbedding(latents.transposed(0, 2, 1).asType(dtype)).asType(.float32)
        for block in blocks {
            x = block(x, context: text, timeModulation: modulation, rotary: rotary, dtype: dtype)
        }
        return head(x, time: time, dtype: dtype).transposed(0, 2, 1)
    }

    /// The checkpoint's tensors under this module's names, from either layout: OpenMOSS's diffusers-style keys
    /// (`condition_embedder.*`, `attn1.to_q`, `ffn.net.0.proj`, `norm2` for the affine norm, `scale_shift_table`…, the
    /// patch embedding a [O, I, 1] convolution) or the MLX pack's native ones (`text_embedding.layers.0`,
    /// `ffn.layers.2`…, [O, 1, I]).
    public static func keys(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        let global: [(String, String)] = [
            ("condition_embedder.text_embedder.linear_1.", "text_embedding_in."),
            ("condition_embedder.text_embedder.linear_2.", "text_embedding_out."),
            ("condition_embedder.time_embedder.linear_1.", "time_embedding_in."),
            ("condition_embedder.time_embedder.linear_2.", "time_embedding_out."),
            ("condition_embedder.time_proj.", "time_projection."),
            ("text_embedding.layers.0.", "text_embedding_in."),
            ("text_embedding.layers.2.", "text_embedding_out."),
            ("time_embedding.layers.0.", "time_embedding_in."),
            ("time_embedding.layers.2.", "time_embedding_out."),
            ("time_projection.layers.1.", "time_projection."),
            ("proj_out.", "head.head."),
        ]
        let block: [(String, String)] = [
            ("attn1.to_out.0.", "self_attn.o."), ("attn2.to_out.0.", "cross_attn.o."),
            ("attn1.to_", "self_attn."), ("attn2.to_", "cross_attn."),
            ("attn1.", "self_attn."), ("attn2.", "cross_attn."),
            ("ffn.net.0.proj.", "ffn_in."), ("ffn.net.2.", "ffn_out."),
            ("ffn.layers.0.", "ffn_in."), ("ffn.layers.2.", "ffn_out."),
            // The affine norm: `norm2` in the export, `norm3` natively (the other two have no parameters).
            ("norm2.", "norm3."),
            ("scale_shift_table", "modulation"),
        ]
        var renamed: [String: MLXArray] = [:]
        for (key, value) in tensors {
            var name = key == "scale_shift_table" ? "head.modulation" : key
            if let (old, new) = global.first(where: { name.hasPrefix($0.0) }) {
                name = new + name.dropFirst(old.count)
            }
            if name.hasPrefix("blocks.") {
                let parts = name.split(separator: ".", maxSplits: 2).map(String.init)
                var suffix = parts[2]
                if let (old, new) = block.first(where: { suffix.hasPrefix($0.0) }) {
                    suffix = new + suffix.dropFirst(old.count)
                }
                name = "blocks.\(parts[1]).\(suffix)"
            }
            var tensor = value
            if name == "patch_embedding.weight", tensor.ndim == 3 {
                tensor = tensor.dim(2) == 1 ? tensor.squeezed(axis: 2) : tensor.squeezed(axis: 1)
            }
            renamed[name] = tensor
        }
        return renamed
    }
}
