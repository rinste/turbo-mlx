import Foundation
import MLX
import MLXNN

// MOSS-Image's DiT, after `MOSSImageTransformer` in OpenMOSS's `pipeline_moss_image.py`: one stream of
// image tokens (row-major over the 1/16 latent grid) followed by the refined text tokens, through 42
// blocks that all read one shared adaLN modulation; the head keeps the image tokens. The dtype points
// are the reference's: every RMSNorm normalizes in float32 and casts back before the weight, the
// timestep sinusoid is float64, RoPE phases float32, the residual stream in the weights' dtype.

/// `_RMSNorm`: `x̂` in float32, cast to x's dtype, then times the weight.
final class MossRMSNorm: Module {
    @ParameterInfo var weight: MLXArray

    init(_ dimensions: Int) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: MLXArray.ones([x.dim(-1)]), eps: 1e-6).asType(x.dtype) * weight
    }
}

/// `_situ`: `(4·tanh(gate/4)·sigmoid(gate)) · (25·tanh(hidden/25))`.
func mossSiTU(_ hidden: MLXArray, _ gate: MLXArray) -> MLXArray {
    (4 * tanh(gate / 4) * sigmoid(gate)) * (25 * tanh(hidden / 25))
}

/// `_attention`: q and k [1, N, dim] rotated per head with `rope` [1, N, headDim] = [sin | cos]
/// (each half duplicated over the head's halves), then attention over every token, unmasked.
func mossAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, rope: MLXArray, heads: Int) -> MLXArray {
    let (batch, length, dim) = (q.dim(0), q.dim(1), q.dim(2))
    let headDim = dim / heads
    let half = rope.dim(-1) / 2
    let sinHalf = rope[.ellipsis, 0 ..< half]
    let cosHalf = rope[.ellipsis, half...]
    let sin = concatenated([sinHalf, sinHalf], axis: -1).expandedDimensions(axis: -2)
    let cos = concatenated([cosHalf, cosHalf], axis: -1).expandedDimensions(axis: -2)
    func rotate(_ x: MLXArray) -> MLXArray {
        let a = x[.ellipsis, 0 ..< (headDim / 2)]
        let b = x[.ellipsis, (headDim / 2)...]
        return x * cos + concatenated([-b, a], axis: -1) * sin
    }
    let shape = [batch, length, heads, headDim]
    let qr = rotate(q.reshaped(shape)).transposed(0, 2, 1, 3).asType(v.dtype)
    let kr = rotate(k.reshaped(shape)).transposed(0, 2, 1, 3).asType(v.dtype)
    let vr = v.reshaped(shape).transposed(0, 2, 1, 3)
    let attended = MLXFast.scaledDotProductAttention(
        queries: qr, keys: kr, values: vr, scale: 1 / Float(headDim).squareRoot(), mask: .none)
    return attended.transposed(0, 2, 1, 3).reshaped([batch, length, dim])
}

/// `_TextBlock`: an unmodulated pre-norm block (biased projections) that refines the prompt.
final class MossTextBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: MossRMSNorm
    @ModuleInfo(key: "norm2") var norm2: MossRMSNorm
    @ModuleInfo(key: "q_norm") var qNorm: MossRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MossRMSNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "ffn_gate") var ffnGate: Linear
    @ModuleInfo(key: "ffn_up") var ffnUp: Linear
    @ModuleInfo(key: "ffn_down") var ffnDown: Linear

    let heads: Int

    init(dim: Int, ffnDim: Int, heads: Int) {
        self.heads = heads
        _norm1.wrappedValue = MossRMSNorm(dim)
        _norm2.wrappedValue = MossRMSNorm(dim)
        _qNorm.wrappedValue = MossRMSNorm(dim / heads)
        _kNorm.wrappedValue = MossRMSNorm(dim / heads)
        _qkv.wrappedValue = Linear(dim, 3 * dim)
        _proj.wrappedValue = Linear(dim, dim)
        _ffnGate.wrappedValue = Linear(dim, ffnDim)
        _ffnUp.wrappedValue = Linear(dim, ffnDim)
        _ffnDown.wrappedValue = Linear(ffnDim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: MLXArray) -> MLXArray {
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2))
        let parts = split(qkv(norm1(x)), parts: 3, axis: -1)
        let q = qNorm(parts[0].reshaped([batch, length, heads, -1])).reshaped([batch, length, dim])
        let k = kNorm(parts[1].reshaped([batch, length, heads, -1])).reshaped([batch, length, dim])
        let h = x + proj(mossAttention(q, k, parts[2], rope: rope, heads: heads))
        let y = norm2(h)
        return h + ffnDown(mossSiTU(ffnUp(y), ffnGate(y)))
    }
}

/// `_TextProjection`: the encoder's states to the model's width, then two text blocks.
final class MossTextProjection: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "blocks") var blocks: [MossTextBlock]

    init(textDim: Int, dim: Int, ffnDim: Int, heads: Int) {
        _proj.wrappedValue = Linear(textDim, dim)
        _blocks.wrappedValue = (0 ..< 2).map { _ in MossTextBlock(dim: dim, ffnDim: ffnDim, heads: heads) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: MLXArray, dtype: DType) -> MLXArray {
        var h = proj(x.asType(dtype))
        for block in blocks { h = block(h, rope: rope) }
        return h
    }
}

/// `_Block.proj`, the block's bias-free projections.
final class MossBlockProjections: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "ffn_up_gate") var ffnUpGate: Linear
    @ModuleInfo(key: "ffn_down") var ffnDown: Linear

    init(dim: Int, ffnDim: Int) {
        _q.wrappedValue = Linear(dim, dim, bias: false)
        _k.wrappedValue = Linear(dim, dim, bias: false)
        _v.wrappedValue = Linear(dim, dim, bias: false)
        _o.wrappedValue = Linear(dim, dim, bias: false)
        _ffnUpGate.wrappedValue = Linear(dim, 2 * ffnDim, bias: false)
        _ffnDown.wrappedValue = Linear(ffnDim, dim, bias: false)
        super.init()
    }
}

/// `_Block`: attention and a SiTU-gated feed-forward, both modulated by the shared adaLN.
final class MossBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: MossRMSNorm
    @ModuleInfo(key: "norm2") var norm2: MossRMSNorm
    @ModuleInfo(key: "q_norm") var qNorm: MossRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MossRMSNorm
    @ModuleInfo(key: "proj") var proj: MossBlockProjections

    let heads: Int

    init(dim: Int, ffnDim: Int, heads: Int) {
        self.heads = heads
        _norm1.wrappedValue = MossRMSNorm(dim)
        _norm2.wrappedValue = MossRMSNorm(dim)
        _qNorm.wrappedValue = MossRMSNorm(dim / heads)
        _kNorm.wrappedValue = MossRMSNorm(dim / heads)
        _proj.wrappedValue = MossBlockProjections(dim: dim, ffnDim: ffnDim)
        super.init()
    }

    /// `modulation` [1, 6, dim]: shift, scale and gate of the attention, then of the feed-forward.
    func callAsFunction(_ x: MLXArray, rope: MLXArray, modulation m: MLXArray, dtype: DType) -> MLXArray {
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2))
        func row(_ index: Int) -> MLXArray { m[0..., index ..< (index + 1), 0...] }
        let y = (norm1(x) * (1 + row(1)) + row(0)).asType(dtype)
        let q = qNorm(proj.q(y).reshaped([batch, length, heads, -1])).reshaped([batch, length, dim])
        let k = kNorm(proj.k(y).reshaped([batch, length, heads, -1])).reshaped([batch, length, dim])
        let h = x + proj.o(mossAttention(q, k, proj.v(y), rope: rope, heads: heads)) * row(2)
        let y2 = norm2(h) * (1 + row(4)) + row(3)
        let parts = split(proj.ffnUpGate(y2.asType(dtype)), parts: 2, axis: -1)
        return h + proj.ffnDown(mossSiTU(parts[0], parts[1])) * row(5)
    }
}

/// `time_embedding`, an `nn.Sequential(Linear, SiLU, Linear)`: the checkpoint's `0` and `2`, renamed
/// on loading (numeric keys would read as list indices).
final class MossTimeEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(freqDim: Int, dim: Int) {
        _linear1.wrappedValue = Linear(freqDim, dim)
        _linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

public final class MossImageTransformer: Module {
    @ModuleInfo(key: "video_embedder") var videoEmbedder: Linear
    @ModuleInfo(key: "text_embedder") var textEmbedder: MossTextProjection
    @ModuleInfo(key: "time_embedding") var timeEmbedding: MossTimeEmbedding
    @ModuleInfo(key: "shared_block_modulation") var sharedBlockModulation: Linear
    @ModuleInfo(key: "layers") var layers: [MossBlock]
    @ModuleInfo(key: "final_norm_video") var finalNormVideo: MossRMSNorm
    @ModuleInfo(key: "final_linear_video") var finalLinearVideo: Linear
    @ModuleInfo(key: "video_head_modulation") var videoHeadModulation: Linear

    public let config: MossImageConfig.Transformer
    /// `_RoPE.bands`, `1 / base^(i / n)` in float32: derived, never loaded (kept out of the
    /// module's parameters as plain numbers).
    let bands: [Float]

    public init(config: MossImageConfig.Transformer) {
        self.config = config
        let count = config.headDim / 8
        bands = (1 / pow(MLXArray(config.ropeBase), MLXArray((0 ..< count).map { Float($0) }) / Float(count))).asArray(Float.self)
        _videoEmbedder.wrappedValue = Linear(config.latentChannels, config.dim)
        _textEmbedder.wrappedValue = MossTextProjection(textDim: config.textDim, dim: config.dim, ffnDim: config.ffnDim, heads: config.numHeads)
        _timeEmbedding.wrappedValue = MossTimeEmbedding(freqDim: config.freqDim, dim: config.dim)
        _sharedBlockModulation.wrappedValue = Linear(config.dim, 6 * config.dim)
        _layers.wrappedValue = (0 ..< config.numLayers).map { _ in MossBlock(dim: config.dim, ffnDim: config.ffnDim, heads: config.numHeads) }
        _finalNormVideo.wrappedValue = MossRMSNorm(config.dim)
        _finalLinearVideo.wrappedValue = Linear(config.dim, config.latentChannels, bias: false)
        _videoHeadModulation.wrappedValue = Linear(config.dim, 2 * config.dim)
        super.init()
    }

    /// The weights' dtype (a norm's: quantized projections pack theirs).
    public var dtype: DType { finalNormVideo.weight.dtype }

    /// The checkpoint's keys under the module's names, minus the derived `rope.bands`.
    public static func keys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in weights where key != "rope.bands" {
            let name = key.replacingOccurrences(of: "time_embedding.0.", with: "time_embedding.linear_1.")
                .replacingOccurrences(of: "time_embedding.2.", with: "time_embedding.linear_2.")
            mapped[name] = value
        }
        return mapped
    }

    /// `_RoPE.forward`: per token the coordinates (r, f, h, w) — the image's (0, text length, row and
    /// column centred on the grid), then the text's (0, index, 0, 0) — times the bands in float32,
    /// laid out [sin of the four axes | cos of the four]: [1, h·w + textLength, headDim], cast.
    public func rope(height: Int, width: Int, textLength: Int, dtype: DType) -> MLXArray {
        let tokens = height * width
        var coordinates = [Float](repeating: 0, count: (tokens + textLength) * 4)
        let (centreRow, centreColumn) = (Float(height - 1) / 2, Float(width - 1) / 2)
        for row in 0 ..< height {
            for column in 0 ..< width {
                let base = (row * width + column) * 4
                coordinates[base + 1] = Float(textLength)
                coordinates[base + 2] = Float(row) - centreRow
                coordinates[base + 3] = Float(column) - centreColumn
            }
        }
        for index in 0 ..< textLength { coordinates[(tokens + index) * 4 + 1] = Float(index) }
        let phase = MLXArray(coordinates, [tokens + textLength, 4, 1]) * MLXArray(bands, [1, 1, bands.count])
        let table = concatenated([MLX.sin(phase), MLX.cos(phase)], axis: -2)
        return table.reshaped([1, tokens + textLength, -1]).asType(dtype)
    }

    /// The shared block modulation [1, 6, dim] and the head's [1, 2, dim] at a timestep: the sinusoid
    /// in float64 (as training computed it), the MLP in the weights' dtype, its output through float32.
    public func modulation(timestep: Float) -> (blocks: MLXArray, head: MLXArray) {
        let half = config.freqDim / 2
        var embedding = [Float](repeating: 0, count: 2 * half)
        for i in 0 ..< half {
            let phase = Double(timestep) * Foundation.pow(10000.0, -Double(i) / Double(half))
            embedding[i] = Float(Foundation.cos(phase))
            embedding[half + i] = Float(Foundation.sin(phase))
        }
        let time = timeEmbedding(MLXArray(embedding, [1, 2 * half]).asType(dtype)).asType(.float32).asType(dtype)
        return (sharedBlockModulation(time).reshaped([1, 6, -1]), videoHeadModulation(time).reshaped([1, 2, -1]))
    }

    /// One forward. `latents` [1, h, w, C] (channels last; any dtype), `context` [1, S, textDim] (the
    /// prompt's real tokens), `skipLayers` the blocks to leave out (guidance's negative branch).
    /// Returns the velocity [1, h, w, C] in the weights' dtype.
    public func callAsFunction(_ latents: MLXArray, timestep: Float, context: MLXArray, skipLayers: Set<Int> = []) -> MLXArray {
        let (height, width, channels) = (latents.dim(1), latents.dim(2), latents.dim(3))
        let tokens = height * width
        let dtype = self.dtype
        let (modulation, head) = modulation(timestep: timestep)
        let video = videoEmbedder(latents.reshaped([1, tokens, channels]).asType(dtype))
        let rope = rope(height: height, width: width, textLength: context.dim(1), dtype: video.dtype)
        let text = textEmbedder(context, rope: rope[0..., tokens..., 0...], dtype: dtype)
        var x = concatenated([video, text], axis: 1)
        for (index, layer) in layers.enumerated() where !skipLayers.contains(index) {
            x = layer(x, rope: rope, modulation: modulation, dtype: dtype)
        }
        var y = finalNormVideo(x[0..., 0 ..< tokens, 0...])
        y = y * (1 + head[0..., 1 ..< 2, 0...]) + head[0..., 0 ..< 1, 0...]
        return finalLinearVideo(y).reshaped([1, height, width, channels])
    }
}
