import Foundation
import MLX
import MLXNN

// The FLUX.2 transformer, after mflux's `flux2_transformer/*`: double-stream blocks over text and
// image, then single-stream blocks over their concatenation. Names follow the reference so the
// checkpoint loads unchanged.

let modelPrecision = DType.bfloat16

/// Rotary tables for the 4-axis ids (t, h, w, token): cos and sin of shape [S, sum(axes)/2].
struct Flux2PosEmbed {
    let theta: Double
    let axesDim: [Int]

    func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        let pos = ids.asType(.float32)
        var cosParts: [MLXArray] = []
        var sinParts: [MLXArray] = []
        for (axis, dim) in axesDim.enumerated() {
            let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
            let omega = 1 / pow(Float(theta), scale)
            let out = pos[.ellipsis, axis].expandedDimensions(axis: -1) * omega.expandedDimensions(axis: 0)
            cosParts.append(cos(out))
            sinParts.append(sin(out))
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }
}

/// Sinusoidal timestep features (256 wide, cos first) through two linears; Klein has no
/// guidance embedding.
final class Flux2TimestepGuidanceEmbeddings: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let inChannels: Int

    init(inChannels: Int, embeddingDim: Int) {
        self.inChannels = inChannels
        _linear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
        _linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        linear2(silu(linear1(Self.features(timestep.asType(.float32), dim: inChannels))))
    }

    static func features(_ timesteps: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let freqs = exp(-Float(log(10000.0)) * MLXArray((0 ..< half).map { Float($0) }) / Float(half))
        let args = timesteps.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        // flip_sin_to_cos: cos, then sin.
        return concatenated([cos(args), sin(args)], axis: -1)
    }
}

/// `silu(temb)` through a linear into `sets` groups of (shift, scale, gate).
final class Flux2Modulation: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let sets: Int

    init(dim: Int, sets: Int) {
        self.sets = sets
        _linear.wrappedValue = Linear(dim, dim * 3 * sets, bias: false)
        super.init()
    }

    /// Each parameter is [B, 1, dim], ready to broadcast over the sequence.
    func callAsFunction(_ temb: MLXArray) -> [[MLXArray]] {
        var mod = linear(silu(temb))
        if mod.ndim == 2 { mod = mod.expandedDimensions(axis: 1) }
        let parts = mod.split(parts: 3 * sets, axis: -1)
        return (0 ..< sets).map { Array(parts[(3 * $0) ..< (3 * $0 + 3)]) }
    }
}

enum Flux2SwiGLU {
    static func apply(_ x: MLXArray) -> MLXArray {
        let halves = x.split(parts: 2, axis: -1)
        return silu(halves[0]) * halves[1]
    }
}

final class Flux2FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(dim: Int, mult: Double) {
        let inner = Int(Double(dim) * mult)
        _linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        _linearOut.wrappedValue = Linear(inner, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearOut(Flux2SwiGLU.apply(linearIn(x)))
    }
}

enum Flux2Rope {
    /// `apply_rope_bshd`: pairs of adjacent channels rotated, in float32, cast back.
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let outType = x.dtype
        let x32 = x.asType(.float32)
        let cosB = cos.reshaped([1, 1, cos.shape[0], cos.shape[1]])
        let sinB = sin.reshaped([1, 1, sin.shape[0], sin.shape[1]])
        let pairs = x32.reshaped(Array(x32.shape.dropLast()) + [x32.shape[x32.ndim - 1] / 2, 2])
        let real = pairs[.ellipsis, 0]
        let imag = pairs[.ellipsis, 1]
        let out0 = real * cosB - imag * sinB
        let out1 = imag * cosB + real * sinB
        return stacked([out0, out1], axis: -1).reshaped(x32.shape).asType(outType)
    }
}

/// q/k/v projections into heads, RMS-normalized q and k in float32 (`AttentionUtils.process_qkv`).
func projectHeads(_ x: MLXArray, q: Linear, k: Linear, v: Linear, normQ: RMSNorm, normK: RMSNorm, heads: Int, headDim: Int) -> (MLXArray, MLXArray, MLXArray) {
    let (batch, length) = (x.shape[0], x.shape[1])
    var query = q(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
    var key = k(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
    let value = v(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
    query = normQ(query.asType(.float32)).asType(x.dtype)
    key = normK(key.asType(.float32)).asType(x.dtype)
    return (query, key, value)
}

/// Attention output back to [B, S, H·D].
func mergeHeads(_ x: MLXArray, heads: Int, headDim: Int) -> MLXArray {
    let batch = x.shape[0]
    return x.transposed(0, 2, 1, 3).reshaped([batch, -1, heads * headDim])
}

/// Joint attention of a double-stream block: text tokens first, then image tokens.
final class Flux2Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    let heads: Int
    let headDim: Int

    init(dim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _toOut.wrappedValue = Linear(inner, dim, bias: false)
        _normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _addQProj.wrappedValue = Linear(dim, inner, bias: false)
        _addKProj.wrappedValue = Linear(dim, inner, bias: false)
        _addVProj.wrappedValue = Linear(dim, inner, bias: false)
        _toAddOut.wrappedValue = Linear(inner, dim, bias: false)
        super.init()
    }

    /// Returns (image output, text output).
    func callAsFunction(image: MLXArray, text: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let (q, k, v) = projectHeads(image, q: toQ, k: toK, v: toV, normQ: normQ, normK: normK, heads: heads, headDim: headDim)
        let (tq, tk, tv) = projectHeads(text, q: addQProj, k: addKProj, v: addVProj, normQ: normAddedQ, normK: normAddedK, heads: heads, headDim: headDim)
        var query = concatenated([tq, q], axis: 2)
        var key = concatenated([tk, k], axis: 2)
        let value = concatenated([tv, v], axis: 2)
        query = Flux2Rope.apply(query, cos: cos, sin: sin)
        key = Flux2Rope.apply(key, cos: cos, sin: sin)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: 1 / Float(headDim).squareRoot(), mask: nil
        )
        let merged = mergeHeads(attended, heads: heads, headDim: headDim)
        let textLength = text.shape[1]
        let textOut = toAddOut(merged[0..., 0 ..< textLength])
        let imageOut = toOut(merged[0..., textLength...])
        return (imageOut, textOut)
    }
}

final class Flux2TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
    @ModuleInfo(key: "attn") var attn: Flux2Attention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ff") var ff: Flux2FeedForward
    @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
    @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Double) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _norm1Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = Flux2Attention(dim: dim, heads: heads, headDim: headDim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _ff.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
        _norm2Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _ffContext.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
        super.init()
    }

    /// `imageMod` / `textMod`: two sets of (shift, scale, gate) each. Returns (text, image).
    func callAsFunction(
        image: MLXArray, text: MLXArray, imageMod: [[MLXArray]], textMod: [[MLXArray]], cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (shiftMsa, scaleMsa, gateMsa) = (imageMod[0][0], imageMod[0][1], imageMod[0][2])
        let (shiftMlp, scaleMlp, gateMlp) = (imageMod[1][0], imageMod[1][1], imageMod[1][2])
        let (cShiftMsa, cScaleMsa, cGateMsa) = (textMod[0][0], textMod[0][1], textMod[0][2])
        let (cShiftMlp, cScaleMlp, cGateMlp) = (textMod[1][0], textMod[1][1], textMod[1][2])

        let normImage = (1 + scaleMsa) * norm1(image) + shiftMsa
        let normText = (1 + cScaleMsa) * norm1Context(text) + cShiftMsa
        let (attnImage, attnText) = attn(image: normImage, text: normText, cos: cos, sin: sin)

        var image = image + gateMsa * attnImage
        var text = text + cGateMsa * attnText

        let normImage2 = (1 + scaleMlp) * norm2(image) + shiftMlp
        image = image + gateMlp * ff(normImage2)
        let normText2 = (1 + cScaleMlp) * norm2Context(text) + cShiftMlp
        text = text + cGateMlp * ffContext(normText2)
        return (text, image)
    }
}

/// One projection feeds attention and the MLP side by side.
final class Flux2ParallelSelfAttention: Module {
    @ModuleInfo(key: "to_qkv_mlp_proj") var toQKVMLP: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear

    let heads: Int
    let headDim: Int
    let innerDim: Int
    let mlpHiddenDim: Int

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Double) {
        self.heads = heads
        self.headDim = headDim
        innerDim = heads * headDim
        mlpHiddenDim = Int(Double(dim) * mlpRatio)
        _toQKVMLP.wrappedValue = Linear(dim, innerDim * 3 + mlpHiddenDim * 2, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _toOut.wrappedValue = Linear(innerDim + mlpHiddenDim, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let projected = toQKVMLP(x)
        let split = projected.split(indices: [innerDim * 3], axis: -1)
        let qkv = split[0].split(parts: 3, axis: -1)
        let mlpHidden = split[1]
        let (batch, length) = (x.shape[0], x.shape[1])
        var query = qkv[0].reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        var key = qkv[1].reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let value = qkv[2].reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        query = normQ(query.asType(.float32)).asType(modelPrecision)
        key = normK(key.asType(.float32)).asType(modelPrecision)
        query = Flux2Rope.apply(query, cos: cos, sin: sin)
        key = Flux2Rope.apply(key, cos: cos, sin: sin)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: 1 / Float(headDim).squareRoot(), mask: nil
        )
        let merged = mergeHeads(attended, heads: heads, headDim: headDim)
        return toOut(concatenated([merged, Flux2SwiGLU.apply(mlpHidden)], axis: -1))
    }
}

final class Flux2SingleTransformerBlock: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "attn") var attn: Flux2ParallelSelfAttention

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Double) {
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = Flux2ParallelSelfAttention(dim: dim, heads: heads, headDim: headDim, mlpRatio: mlpRatio)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mod: [MLXArray], cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (shift, scale, gate) = (mod[0], mod[1], mod[2])
        let normed = (1 + scale) * norm(x) + shift
        return x + gate * attn(normed, cos: cos, sin: sin)
    }
}

/// The final norm, modulated by the timestep embedding.
final class AdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let embeddingDim: Int

    init(embeddingDim: Int, conditioningDim: Int) {
        self.embeddingDim = embeddingDim
        _linear.wrappedValue = Linear(conditioningDim, embeddingDim * 2, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        let modulation = linear(silu(temb).asType(modelPrecision))
        let scale = modulation[0..., 0 ..< embeddingDim]
        let shift = modulation[0..., embeddingDim ..< (2 * embeddingDim)]
        return norm(x) * (1 + scale).expandedDimensions(axis: 1) + shift.expandedDimensions(axis: 1)
    }
}

public final class Flux2Transformer: Module {
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2TimestepGuidanceEmbeddings
    @ModuleInfo(key: "double_stream_modulation_img") var doubleStreamModulationImg: Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleStreamModulationTxt: Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var singleStreamModulation: Flux2Modulation
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux2SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let config: KleinConfig.Transformer
    let posEmbed: Flux2PosEmbed

    public init(config: KleinConfig.Transformer) {
        self.config = config
        let inner = config.innerDim
        posEmbed = Flux2PosEmbed(theta: config.ropeTheta, axesDim: config.ropeAxesDim)
        _timeGuidanceEmbed.wrappedValue = Flux2TimestepGuidanceEmbeddings(inChannels: config.timestepGuidanceChannels, embeddingDim: inner)
        _doubleStreamModulationImg.wrappedValue = Flux2Modulation(dim: inner, sets: 2)
        _doubleStreamModulationTxt.wrappedValue = Flux2Modulation(dim: inner, sets: 2)
        _singleStreamModulation.wrappedValue = Flux2Modulation(dim: inner, sets: 1)
        _xEmbedder.wrappedValue = Linear(config.inChannels, inner, bias: false)
        _contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, inner, bias: false)
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            Flux2TransformerBlock(dim: inner, heads: config.numAttentionHeads, headDim: config.attentionHeadDim, mlpRatio: config.mlpRatio)
        }
        _singleTransformerBlocks.wrappedValue = (0 ..< config.numSingleLayers).map { _ in
            Flux2SingleTransformerBlock(dim: inner, heads: config.numAttentionHeads, headDim: config.attentionHeadDim, mlpRatio: config.mlpRatio)
        }
        _normOut.wrappedValue = AdaLayerNormContinuous(embeddingDim: inner, conditioningDim: inner)
        _projOut.wrappedValue = Linear(inner, config.inChannels, bias: false)
        super.init()
    }

    /// One denoising pass. `latents` [B, S, 128] packed, `prompt` [B, T, joint dim], ids [S, 4] and
    /// [T, 4] (a batch axis is dropped), `timestep` a scalar in the scheduler's units.
    public func callAsFunction(
        latents: MLXArray, prompt: MLXArray, timestep: Float, imageIds: MLXArray, textIds: MLXArray
    ) -> MLXArray {
        let batch = latents.shape[0]
        // As the reference: the timestep goes to the activations' dtype first, and one at or below 1
        // is taken to be in [0, 1] and scaled to [0, 1000] there.
        var timesteps = MLXArray(Array(repeating: timestep, count: batch)).asType(latents.dtype)
        if timestep <= 1 { timesteps = timesteps * 1000 }
        let temb = timeGuidanceEmbed(timesteps).asType(modelPrecision)

        var image = xEmbedder(latents)
        var text = contextEmbedder(prompt)
        let imageIds2D = imageIds.ndim == 3 ? imageIds[0] : imageIds
        let textIds2D = textIds.ndim == 3 ? textIds[0] : textIds
        let (imageCos, imageSin) = posEmbed(imageIds2D)
        let (textCos, textSin) = posEmbed(textIds2D)
        let cos = concatenated([textCos, imageCos], axis: 0)
        let sin = concatenated([textSin, imageSin], axis: 0)

        let imageMod = doubleStreamModulationImg(temb)
        let textMod = doubleStreamModulationTxt(temb)
        for block in transformerBlocks {
            (text, image) = block(image: image, text: text, imageMod: imageMod, textMod: textMod, cos: cos, sin: sin)
        }

        var hidden = concatenated([text, image], axis: 1)
        let singleMod = singleStreamModulation(temb)[0]
        for block in singleTransformerBlocks {
            hidden = block(hidden, mod: singleMod, cos: cos, sin: sin)
        }
        hidden = hidden[0..., text.shape[1]...]
        return projOut(normOut(hidden, temb: temb))
    }
}
