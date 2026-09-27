import Foundation
import MLX
import MLXNN

// Qwen-Image's transformer, after mflux's `qwen_transformer/*`: 60 dual-stream blocks over text
// and image tokens with joint attention. Names follow the reference so the checkpoint loads
// unchanged. mflux creates the latents in float32 and never casts them, so the image stream (and,
// from the first block on, the text stream) runs in float32; this port does the same by using the
// same operations without casts of its own.

/// `QwenTransformerRMSNorm`: the variance in float32, the weight applied in its own dtype.
final class QwenTransformerRMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputType = x.dtype
        let variance = square(x.asType(.float32)).mean(axis: -1, keepDims: true)
        var h = x * rsqrt(variance + eps)
        if weight.dtype == .bfloat16 || weight.dtype == .float16 {
            h = h.asType(weight.dtype)
        }
        h = h * weight
        return h.dtype == inputType ? h : h.asType(inputType)
    }
}

final class QwenTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(projDim: Int, innerDim: Int) {
        _linear1.wrappedValue = Linear(projDim, innerDim, bias: true)
        _linear2.wrappedValue = Linear(innerDim, innerDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

/// `QwenTimeTextEmbed`: sinusoidal features of the timestep (× 1000, cos first) through the embedder.
final class QwenTimeTextEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: QwenTimestepEmbedding
    let projDim: Int

    init(projDim: Int = 256, innerDim: Int) {
        self.projDim = projDim
        _timestepEmbedder.wrappedValue = QwenTimestepEmbedding(projDim: projDim, innerDim: innerDim)
        super.init()
    }

    /// `timestep` [B] in the activations' dtype.
    func callAsFunction(_ timestep: MLXArray, hiddenType: DType) -> MLXArray {
        timestepEmbedder(features(timestep).asType(hiddenType))
    }

    /// `QwenTimesteps`: `[sin, cos]` of `1000 · t · freqs`, flipped to cos first.
    func features(_ timesteps: MLXArray) -> MLXArray {
        let half = projDim / 2
        let exponent = -Float(log(10000.0)) * MLXArray((0 ..< half).map { Float($0) }) / Float(half)
        let freqs = exp(exponent).asType(timesteps.dtype)
        var emb = timesteps.expandedDimensions(axis: 1).asType(.float32) * freqs.expandedDimensions(axis: 0)
        emb = 1000 * emb
        let sinCos = concatenated([sin(emb), cos(emb)], axis: -1)
        return concatenated([sinCos[0..., half...], sinCos[0..., 0 ..< half]], axis: -1)
    }
}

/// Joint attention of a block: text tokens first, then image tokens.
final class QwenImageAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "attn_to_out") var attnToOut: [Linear]
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    let heads: Int
    let headDim: Int

    init(dim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        _toQ.wrappedValue = Linear(dim, dim)
        _toK.wrappedValue = Linear(dim, dim)
        _toV.wrappedValue = Linear(dim, dim)
        _addQProj.wrappedValue = Linear(dim, dim)
        _addKProj.wrappedValue = Linear(dim, dim)
        _addVProj.wrappedValue = Linear(dim, dim)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _attnToOut.wrappedValue = [Linear(dim, dim)]
        _toAddOut.wrappedValue = Linear(dim, dim)
        super.init()
    }

    /// Returns (image output, text output).
    func callAsFunction(image: MLXArray, text: MLXArray, imageCos: MLXArray, imageSin: MLXArray, textCos: MLXArray, textSin: MLXArray) -> (MLXArray, MLXArray) {
        let (batch, imageLength, textLength) = (image.shape[0], image.shape[1], text.shape[1])
        var q = normQ(toQ(image).reshaped([batch, imageLength, heads, headDim]))
        var k = normK(toK(image).reshaped([batch, imageLength, heads, headDim]))
        let v = toV(image).reshaped([batch, imageLength, heads, headDim])
        var tq = normAddedQ(addQProj(text).reshaped([batch, textLength, heads, headDim]))
        var tk = normAddedK(addKProj(text).reshaped([batch, textLength, heads, headDim]))
        let tv = addVProj(text).reshaped([batch, textLength, heads, headDim])

        q = S3DiTRope.apply(q, cos: imageCos, sin: imageSin, castBack: true)
        k = S3DiTRope.apply(k, cos: imageCos, sin: imageSin, castBack: true)
        tq = S3DiTRope.apply(tq, cos: textCos, sin: textSin, castBack: true)
        tk = S3DiTRope.apply(tk, cos: textCos, sin: textSin, castBack: true)

        let query = concatenated([tq, q], axis: 1).transposed(0, 2, 1, 3)
        let key = concatenated([tk, k], axis: 1).transposed(0, 2, 1, 3)
        let value = concatenated([tv, v], axis: 1).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value, scale: 1 / Float(headDim).squareRoot(), mask: nil
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, textLength + imageLength, heads * headDim]).asType(query.dtype)
        let textOut = toAddOut(merged[0..., 0 ..< textLength])
        let imageOut = attnToOut[0](merged[0..., textLength...])
        return (imageOut, textOut)
    }
}

final class QwenFeedForward: Module {
    @ModuleInfo(key: "mlp_in") var mlpIn: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    init(dim: Int) {
        _mlpIn.wrappedValue = Linear(dim, 4 * dim, bias: true)
        _mlpOut.wrappedValue = Linear(4 * dim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlpOut(geluApproximate(mlpIn(x)))
    }
}

final class QwenImageTransformerBlock: Module {
    @ModuleInfo(key: "img_mod_linear") var imgModLinear: Linear
    @ModuleInfo(key: "img_norm1") var imgNorm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: QwenImageAttention
    @ModuleInfo(key: "img_norm2") var imgNorm2: LayerNorm
    @ModuleInfo(key: "img_ff") var imgFF: QwenFeedForward
    @ModuleInfo(key: "txt_mod_linear") var txtModLinear: Linear
    @ModuleInfo(key: "txt_norm1") var txtNorm1: LayerNorm
    @ModuleInfo(key: "txt_norm2") var txtNorm2: LayerNorm
    @ModuleInfo(key: "txt_ff") var txtFF: QwenFeedForward

    init(dim: Int, heads: Int, headDim: Int) {
        _imgModLinear.wrappedValue = Linear(dim, 6 * dim, bias: true)
        _imgNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = QwenImageAttention(dim: dim, heads: heads, headDim: headDim)
        _imgNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _imgFF.wrappedValue = QwenFeedForward(dim: dim)
        _txtModLinear.wrappedValue = Linear(dim, 6 * dim, bias: true)
        _txtNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _txtNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _txtFF.wrappedValue = QwenFeedForward(dim: dim)
        super.init()
    }

    /// Returns (text, image).
    func callAsFunction(
        image: MLXArray, text: MLXArray, temb: MLXArray,
        imageCos: MLXArray, imageSin: MLXArray, textCos: MLXArray, textSin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let imgMod = imgModLinear(silu(temb)).split(parts: 2, axis: -1)
        let txtMod = txtModLinear(silu(temb)).split(parts: 2, axis: -1)

        let (imgModulated, imgGate1) = Self.modulate(imgNorm1(image), imgMod[0])
        let (txtModulated, txtGate1) = Self.modulate(txtNorm1(text), txtMod[0])
        let (imgAttn, txtAttn) = attn(image: imgModulated, text: txtModulated, imageCos: imageCos, imageSin: imageSin, textCos: textCos, textSin: textSin)
        var image = image + imgGate1 * imgAttn
        var text = text + txtGate1 * txtAttn

        let (imgModulated2, imgGate2) = Self.modulate(imgNorm2(image), imgMod[1])
        image = image + imgGate2 * imgFF(imgModulated2)
        let (txtModulated2, txtGate2) = Self.modulate(txtNorm2(text), txtMod[1])
        text = text + txtGate2 * txtFF(txtModulated2)
        return (text, image)
    }

    /// (shift, scale, gate) from a modulation row: `x · (1 + scale) + shift`, and the gate.
    static func modulate(_ x: MLXArray, _ params: MLXArray) -> (MLXArray, MLXArray) {
        let parts = params.split(parts: 3, axis: -1)
        let (shift, scale, gate) = (parts[0], parts[1], parts[2])
        return (x * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1), gate.expandedDimensions(axis: 1))
    }
}

/// `QwenEmbedRope` with `scale_rope`: image positions centred on each axis, text positions after
/// the larger half-side; cos and sin over the axes' frequencies, [S, sum(axes)/2] in float32.
struct QwenImageRope {
    let theta: Float
    let axesDim: [Int]

    private func table(positions: [Float], dim: Int) -> (MLXArray, MLXArray) {
        let scales = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let omega = 1 / pow(theta, scales)
        let freqs = outer(MLXArray(positions), omega)
        return (cos(freqs), sin(freqs))
    }

    /// The image grid `height` × `width` (one frame) and `textLength` text tokens.
    func callAsFunction(height: Int, width: Int, textLength: Int) -> (imageCos: MLXArray, imageSin: MLXArray, textCos: MLXArray, textSin: MLXArray) {
        // Frame axis: position 0 for every token; height and width: from −(n − n/2) to n/2 − 1.
        let (frameCos, frameSin) = table(positions: [0], dim: axesDim[0])
        let (heightCos, heightSin) = table(positions: (0 ..< height).map { Float($0 - (height - height / 2)) }, dim: axesDim[1])
        let (widthCos, widthSin) = table(positions: (0 ..< width).map { Float($0 - (width - width / 2)) }, dim: axesDim[2])
        let count = height * width
        let frameC = broadcast(frameCos, to: [count, axesDim[0] / 2])
        let frameS = broadcast(frameSin, to: [count, axesDim[0] / 2])
        let heightC = broadcast(heightCos.reshaped([height, 1, axesDim[1] / 2]), to: [height, width, axesDim[1] / 2]).reshaped([count, axesDim[1] / 2])
        let heightS = broadcast(heightSin.reshaped([height, 1, axesDim[1] / 2]), to: [height, width, axesDim[1] / 2]).reshaped([count, axesDim[1] / 2])
        let widthC = broadcast(widthCos.reshaped([1, width, axesDim[2] / 2]), to: [height, width, axesDim[2] / 2]).reshaped([count, axesDim[2] / 2])
        let widthS = broadcast(widthSin.reshaped([1, width, axesDim[2] / 2]), to: [height, width, axesDim[2] / 2]).reshaped([count, axesDim[2] / 2])
        let imageCos = concatenated([frameC, heightC, widthC], axis: -1)
        let imageSin = concatenated([frameS, heightS, widthS], axis: -1)

        // Text: the same position on all three axes, from max(h/2, w/2).
        let start = max(height / 2, width / 2)
        let positions = (0 ..< textLength).map { Float(start + $0) }
        var textCosParts: [MLXArray] = []
        var textSinParts: [MLXArray] = []
        for dim in axesDim {
            let (c, s) = table(positions: positions, dim: dim)
            textCosParts.append(c)
            textSinParts.append(s)
        }
        return (imageCos, imageSin, concatenated(textCosParts, axis: -1), concatenated(textSinParts, axis: -1))
    }
}

public final class QwenImageTransformer: Module {
    @ModuleInfo(key: "img_in") var imgIn: Linear
    @ModuleInfo(key: "txt_norm") var txtNorm: QwenTransformerRMSNorm
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: QwenTimeTextEmbed
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [QwenImageTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public let config: QwenImageConfig.Transformer
    let rope: QwenImageRope

    public init(config: QwenImageConfig.Transformer) {
        self.config = config
        let inner = config.innerDim
        rope = QwenImageRope(theta: config.ropeTheta, axesDim: config.ropeAxesDim)
        _imgIn.wrappedValue = Linear(config.inChannels, inner)
        _txtNorm.wrappedValue = QwenTransformerRMSNorm(dimensions: config.jointAttentionDim, eps: 1e-6)
        _txtIn.wrappedValue = Linear(config.jointAttentionDim, inner)
        _timeTextEmbed.wrappedValue = QwenTimeTextEmbed(projDim: 256, innerDim: inner)
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            QwenImageTransformerBlock(dim: inner, heads: config.numAttentionHeads, headDim: config.attentionHeadDim)
        }
        _normOut.wrappedValue = AdaLayerNormContinuous(embeddingDim: inner, conditioningDim: inner)
        _projOut.wrappedValue = Linear(inner, Self.patchSize * Self.patchSize * config.outChannels)
        super.init()
    }

    static var patchSize: Int { QwenImageConfig.Transformer.patchSize }

    /// One pass. `latents` [B, h·w, 64] packed (float32, as mflux creates them), `prompt`
    /// [B, T, joint dim] bf16, `timestep` the step's sigma in [0, 1], the latent grid `h` × `w`.
    public func callAsFunction(latents: MLXArray, prompt: MLXArray, timestep: Float, latentHeight: Int, latentWidth: Int) -> MLXArray {
        var hidden = imgIn(latents)
        let batch = hidden.shape[0]
        let timesteps = MLXArray(Array(repeating: timestep, count: batch)).asType(hidden.dtype)
        var text = txtIn(txtNorm(prompt))
        let temb = timeTextEmbed(timesteps, hiddenType: hidden.dtype)
        let ropes = rope(height: latentHeight, width: latentWidth, textLength: prompt.shape[1])
        for block in transformerBlocks {
            (text, hidden) = block(
                image: hidden, text: text, temb: temb,
                imageCos: ropes.imageCos, imageSin: ropes.imageSin, textCos: ropes.textCos, textSin: ropes.textSin
            )
        }
        return projOut(normOut(hidden, temb: temb))
    }
}
