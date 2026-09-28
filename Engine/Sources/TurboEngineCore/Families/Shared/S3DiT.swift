import Foundation
import MLX
import MLXNN

// The single-stream DiT that Z-Image and Ming-Image share ("S3-DiT", after mflux's
// `z_image_transformer/*` and `ming_transformer.py`): a noise refiner over the image tokens and a
// context refiner over the caption tokens, then the main layers over their concatenation, each
// block modulated by the timestep. Module and parameter names follow mflux, so both families'
// checkpoints load unchanged. The two differ in what they do around the blocks (see the config).

public struct S3DiTConfig: Equatable, Sendable {
    public var dim = 3840
    public var numLayers = 30
    public var numRefinerLayers = 2
    public var numHeads = 30
    public var normEps: Float = 1e-5
    public var qkNormEps: Float = 1e-5
    public var capFeatDim = 2560
    public var ropeTheta: Float = 256
    public var tScale: Float = 1000
    public var axesDims = [32, 48, 48]
    public var axesLens = [1024, 512, 512]
    public var inChannels = 16
    public var frequencyEmbeddingSize = 256
    /// Z-Image pads the caption and the image tokens to a multiple of 32 with learned pad tokens;
    /// Ming drops the padding (its image positions still count the padded caption length).
    public var padsToMultiple = true
    /// Ming casts the timestep embedding, the latents and the captions to the weights' dtype and
    /// applies its rotary embedding in float32 before casting back; Z-Image casts nothing, and
    /// its float32 timestep embedding promotes the residual stream to float32.
    public var keepsWeightsPrecision = false

    public static let patchSize = 2
    public var headDim: Int { dim / numHeads }
    public var embedDim: Int { Self.patchSize * Self.patchSize * inChannels }
    public var tEmbedSize: Int { min(dim, 256) }
    /// `int(dim / 3 * 8)`.
    public var ffnHidden: Int { Int(Double(dim) / 3 * 8) }

    public init() {}
}

/// Rotary tables for the 3-axis ids (t, h, w) up to `axesLens`: cos and sin of shape [S, sum(axes)/2].
struct S3DiTRopeEmbedder {
    let cosTables: [MLXArray]
    let sinTables: [MLXArray]

    init(theta: Float, axesDims: [Int], axesLens: [Int]) {
        var cosTables: [MLXArray] = []
        var sinTables: [MLXArray] = []
        for (dim, length) in zip(axesDims, axesLens) {
            let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
            let freqs = 1 / pow(theta, exponents)
            let positions = MLXArray((0 ..< length).map { Float($0) })
            let angles = outer(positions, freqs)
            cosTables.append(cos(angles))
            sinTables.append(sin(angles))
        }
        // Plain arrays outside the module tree: materialize them once.
        eval(cosTables + sinTables)
        self.cosTables = cosTables
        self.sinTables = sinTables
    }

    /// `ids` [S, 3] int32 → (cos, sin) each [S, sum(axes)/2].
    func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        var cosParts: [MLXArray] = []
        var sinParts: [MLXArray] = []
        for axis in 0 ..< cosTables.count {
            let index = ids[.ellipsis, axis].asType(.int32)
            cosParts.append(cosTables[axis].take(index, axis: 0))
            sinParts.append(sinTables[axis].take(index, axis: 0))
        }
        return (concatenated(cosParts, axis: 1), concatenated(sinParts, axis: 1))
    }
}

enum S3DiTRope {
    /// Pairs of adjacent channels rotated; `x` [B, S, H, D], cos/sin [S, D/2]. With `castBack`
    /// (Ming) the rotation runs in float32 and returns in x's dtype; without it (Z-Image) the
    /// float32 tables promote a bf16 input to float32, as they do in mflux.
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray, castBack: Bool) -> MLXArray {
        let input = castBack ? x.asType(.float32) : x
        let shape = input.shape
        let pairs = input.reshaped([shape[0], shape[1], shape[2], shape[3] / 2, 2])
        let real = pairs[.ellipsis, 0]
        let imag = pairs[.ellipsis, 1]
        let cosB = cos.reshaped([1, cos.shape[0], 1, cos.shape[1]])
        let sinB = sin.reshaped([1, sin.shape[0], 1, sin.shape[1]])
        let outReal = real * cosB - imag * sinB
        let outImag = real * sinB + imag * cosB
        let rotated = stacked([outReal, outImag], axis: -1).reshaped(shape)
        return castBack ? rotated.asType(x.dtype) : rotated
    }
}

final class S3DiTAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let heads: Int
    let headDim: Int
    let castsRope: Bool

    init(config: S3DiTConfig) {
        heads = config.numHeads
        headDim = config.headDim
        castsRope = config.keepsWeightsPrecision
        _toQ.wrappedValue = Linear(config.dim, config.dim, bias: false)
        _toK.wrappedValue = Linear(config.dim, config.dim, bias: false)
        _toV.wrappedValue = Linear(config.dim, config.dim, bias: false)
        _toOut.wrappedValue = [Linear(config.dim, config.dim, bias: false)]
        _normQ.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.qkNormEps)
        _normK.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.qkNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        var q = normQ(toQ(x).reshaped([batch, length, heads, headDim]))
        var k = normK(toK(x).reshaped([batch, length, heads, headDim]))
        let v = toV(x).reshaped([batch, length, heads, headDim])
        q = S3DiTRope.apply(q, cos: cos, sin: sin, castBack: castsRope).transposed(0, 2, 1, 3)
        k = S3DiTRope.apply(k, cos: cos, sin: sin, castBack: castsRope).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v.transposed(0, 2, 1, 3), scale: 1 / Float(headDim).squareRoot(), mask: nil
        )
        return toOut[0](attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }
}

final class S3DiTFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

/// A transformer block; with `modulated`, the timestep scales the normalized inputs and gates the
/// outputs (`adaLN_modulation`), otherwise it is a plain pre-norm block (the context refiner).
final class S3DiTBlock: Module {
    @ModuleInfo(key: "attention") var attention: S3DiTAttention
    @ModuleInfo(key: "feed_forward") var feedForward: S3DiTFeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLN: [Linear]

    init(config: S3DiTConfig, modulated: Bool) {
        _attention.wrappedValue = S3DiTAttention(config: config)
        _feedForward.wrappedValue = S3DiTFeedForward(dim: config.dim, hiddenDim: config.ffnHidden)
        _attentionNorm1.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _attentionNorm2.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ffnNorm1.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ffnNorm2.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _adaLN.wrappedValue = modulated ? [Linear(config.tEmbedSize, 4 * config.dim, bias: true)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, tEmb: MLXArray?) -> MLXArray {
        guard let tEmb, let modulation = adaLN.first else {
            var h = x + attentionNorm2(attention(attentionNorm1(x), cos: cos, sin: sin))
            h = h + ffnNorm2(feedForward(ffnNorm1(h)))
            return h
        }
        let parts = modulation(tEmb).expandedDimensions(axis: 1).split(parts: 4, axis: 2)
        let (scaleMsa, gateMsa, scaleMlp, gateMlp) = (1 + parts[0], tanh(parts[1]), 1 + parts[2], tanh(parts[3]))
        var h = x + gateMsa * attentionNorm2(attention(attentionNorm1(x) * scaleMsa, cos: cos, sin: sin))
        h = h + gateMlp * ffnNorm2(feedForward(ffnNorm1(h) * scaleMlp))
        return h
    }
}

/// Sinusoidal timestep features (cos first) through two linears.
final class S3DiTTimestepEmbedder: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    let frequencyEmbeddingSize: Int

    init(outSize: Int, midSize: Int, frequencyEmbeddingSize: Int) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        _linear1.wrappedValue = Linear(frequencyEmbeddingSize, midSize, bias: true)
        _linear2.wrappedValue = Linear(midSize, outSize, bias: true)
        super.init()
    }

    /// `t` [B] float32 (already scaled by `t_scale`).
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = frequencyEmbeddingSize / 2
        let freqs = exp(-Float(log(10000.0)) * MLXArray((0 ..< half).map { Float($0) }) / Float(half))
        let args = t.expandedDimensions(axis: 1).asType(.float32) * freqs.expandedDimensions(axis: 0)
        let embedding = concatenated([cos(args), sin(args)], axis: -1)
        return linear2(silu(linear1(embedding)))
    }
}

final class S3DiTFinalLayer: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLN: [Linear]

    init(hiddenSize: Int, outChannels: Int, tEmbedSize: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
        _linear.wrappedValue = Linear(hiddenSize, outChannels, bias: true)
        _adaLN.wrappedValue = [Linear(tEmbedSize, hiddenSize, bias: true)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let scale = (1 + adaLN[0](silu(c))).expandedDimensions(axis: 1)
        return linear(norm(x) * scale)
    }
}

/// mflux keeps the patch embedder and the final layer in dicts keyed by "patch-frame patch" ("2-1").
final class S3DiTPatchEmbedder: Module {
    @ModuleInfo(key: "2-1") var linear: Linear
    init(embedDim: Int, dim: Int) {
        _linear.wrappedValue = Linear(embedDim, dim, bias: true)
        super.init()
    }
}

final class S3DiTFinalLayers: Module {
    @ModuleInfo(key: "2-1") var layer: S3DiTFinalLayer
    init(dim: Int, embedDim: Int, tEmbedSize: Int) {
        _layer.wrappedValue = S3DiTFinalLayer(hiddenSize: dim, outChannels: embedDim, tEmbedSize: tEmbedSize)
        super.init()
    }
}

public final class S3DiTTransformer: Module {
    @ModuleInfo(key: "all_x_embedder") var xEmbedder: S3DiTPatchEmbedder
    @ModuleInfo(key: "all_final_layer") var finalLayers: S3DiTFinalLayers
    @ModuleInfo(key: "t_embedder") var tEmbedder: S3DiTTimestepEmbedder
    /// mflux's `cap_embedder` list: an RMSNorm then a linear.
    @ModuleInfo(key: "cap_embedder") var capEmbedder: [UnaryLayer]
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [S3DiTBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [S3DiTBlock]
    @ModuleInfo(key: "layers") var layers: [S3DiTBlock]
    /// Z-Image's learned pad tokens (absent from Ming's checkpoint).
    var x_pad_token: MLXArray?
    var cap_pad_token: MLXArray?

    public let config: S3DiTConfig
    let ropeEmbedder: S3DiTRopeEmbedder

    public init(config: S3DiTConfig) {
        self.config = config
        ropeEmbedder = S3DiTRopeEmbedder(theta: config.ropeTheta, axesDims: config.axesDims, axesLens: config.axesLens)
        _xEmbedder.wrappedValue = S3DiTPatchEmbedder(embedDim: config.embedDim, dim: config.dim)
        _finalLayers.wrappedValue = S3DiTFinalLayers(dim: config.dim, embedDim: config.embedDim, tEmbedSize: config.tEmbedSize)
        _tEmbedder.wrappedValue = S3DiTTimestepEmbedder(outSize: config.tEmbedSize, midSize: 1024, frequencyEmbeddingSize: config.frequencyEmbeddingSize)
        _capEmbedder.wrappedValue = [RMSNorm(dimensions: config.capFeatDim, eps: config.normEps), Linear(config.capFeatDim, config.dim, bias: true)]
        _noiseRefiner.wrappedValue = (0 ..< config.numRefinerLayers).map { _ in S3DiTBlock(config: config, modulated: true) }
        _contextRefiner.wrappedValue = (0 ..< config.numRefinerLayers).map { _ in S3DiTBlock(config: config, modulated: false) }
        _layers.wrappedValue = (0 ..< config.numLayers).map { _ in S3DiTBlock(config: config, modulated: true) }
        if config.padsToMultiple {
            x_pad_token = MLXArray.zeros([1, config.dim])
            cap_pad_token = MLXArray.zeros([1, config.dim])
        }
        super.init()
    }

    /// One pass. `latents` [C, 1, H, W] (channels first, as mflux keeps them), `timestep` [1] in
    /// [0, 1] (`1 − σ`), `capFeats` [N, capFeatDim] through the caption embedder, `extraCaption`
    /// [N2, dim] appended as it is (Ming's direct-VLM tokens). Returns the flow prediction
    /// [C, 1, H, W] with mflux's sign (negated).
    public func callAsFunction(latents: MLXArray, timestep: MLXArray, capFeats: MLXArray, extraCaption: MLXArray? = nil) -> MLXArray {
        let keeps = config.keepsWeightsPrecision
        var tEmb = tEmbedder(timestep.asType(.float32) * config.tScale)
        if keeps { tEmb = tEmb.asType(modelPrecision) }

        let (tokens, size) = Self.patchify(keeps ? latents.asType(modelPrecision) : latents)
        let imageCount = tokens.shape[0]
        var caption = capEmbedder[1](capEmbedder[0](keeps ? capFeats.asType(modelPrecision) : capFeats))
        if let extraCaption {
            caption = concatenated([caption, extraCaption.asType(modelPrecision)], axis: 0)
        }
        let capCount = caption.shape[0]
        let capPadded = capCount + (32 - capCount % 32) % 32

        // Positions: the caption on the t axis from 1, the image grid after the (32-aligned) caption.
        var capIds = Self.coordinates([capPadded, 1, 1], start: [1, 0, 0])
        var imageIds = Self.coordinates([size.frames, size.height / S3DiTConfig.patchSize, size.width / S3DiTConfig.patchSize], start: [capPadded + 1, 0, 0])
        var image = tokens
        var imagePad = 0
        if config.padsToMultiple {
            let capPad = capPadded - capCount
            if capPad > 0 {
                caption = concatenated([caption, repeated(caption[(capCount - 1)...], count: capPad, axis: 0)], axis: 0)
            }
            imagePad = (32 - imageCount % 32) % 32
            if imagePad > 0 {
                imageIds = concatenated([imageIds, MLXArray.zeros([imagePad, 3], dtype: .int32)], axis: 0)
                image = concatenated([image, repeated(image[(imageCount - 1)...], count: imagePad, axis: 0)], axis: 0)
            }
        } else {
            capIds = capIds[0 ..< capCount]
        }
        let (imageCos, imageSin) = ropeEmbedder(imageIds)
        let (capCos, capSin) = ropeEmbedder(capIds)

        var img = xEmbedder.linear(image)
        if config.padsToMultiple, let x_pad_token {
            img = MLX.where(Self.padMask(real: imageCount, padded: imagePad), x_pad_token, img)
        }
        img = img.expandedDimensions(axis: 0)
        for block in noiseRefiner {
            img = block(img, cos: imageCos, sin: imageSin, tEmb: tEmb)
        }

        var cap = caption
        if config.padsToMultiple, let cap_pad_token {
            cap = MLX.where(Self.padMask(real: capCount, padded: capPadded - capCount), cap_pad_token, cap)
        }
        cap = cap.expandedDimensions(axis: 0)
        for block in contextRefiner {
            cap = block(cap, cos: capCos, sin: capSin, tEmb: nil)
        }

        let imageLength = img.shape[1]
        var unified = concatenated([img, cap], axis: 1)
        let cos = concatenated([imageCos, capCos], axis: 0)
        let sin = concatenated([imageSin, capSin], axis: 0)
        for layer in layers {
            unified = layer(unified, cos: cos, sin: sin, tEmb: tEmb)
        }
        let output = finalLayers.layer(unified[0..., 0 ..< imageLength], c: tEmb)[0]
        return -Self.unpatchify(output[0 ..< imageCount], size: size, outChannels: config.inChannels)
    }

    struct LatentSize {
        let frames: Int
        let height: Int
        let width: Int
    }

    /// [C, F, H, W] → [F·(H/p)·(W/p), p·p·C] tokens, (p, p, C) inside each, row-major over the grid.
    static func patchify(_ x: MLXArray) -> (MLXArray, LatentSize) {
        let p = S3DiTConfig.patchSize
        let (c, f, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let reshaped = x.reshaped([c, f, 1, h / p, p, w / p, p])
        let ordered = reshaped.transposed(1, 3, 5, 2, 4, 6, 0)
        return (ordered.reshaped([f * (h / p) * (w / p), p * p * c]), LatentSize(frames: f, height: h, width: w))
    }

    /// The inverse: tokens [F·(H/p)·(W/p), p·p·C] → [C, F, H, W].
    static func unpatchify(_ x: MLXArray, size: LatentSize, outChannels: Int) -> MLXArray {
        let p = S3DiTConfig.patchSize
        let (f, h, w) = (size.frames, size.height, size.width)
        let reshaped = x.reshaped([f, h / p, w / p, 1, p, p, outChannels])
        return reshaped.transposed(6, 0, 3, 1, 4, 2, 5).reshaped([outChannels, f, h, w])
    }

    /// `_create_coord_grid`: rows of (t, h, w) over a grid, row-major, from `start`.
    static func coordinates(_ size: [Int], start: [Int]) -> MLXArray {
        var values: [Int32] = []
        values.reserveCapacity(size[0] * size[1] * size[2] * 3)
        for t in 0 ..< size[0] {
            for h in 0 ..< size[1] {
                for w in 0 ..< size[2] {
                    values.append(contentsOf: [Int32(start[0] + t), Int32(start[1] + h), Int32(start[2] + w)])
                }
            }
        }
        return MLXArray(values, [size[0] * size[1] * size[2], 3])
    }

    /// [real + padded, 1] true for the padding rows.
    static func padMask(real: Int, padded: Int) -> MLXArray {
        let values = (0 ..< (real + padded)).map { $0 >= real }
        return MLXArray(values, [real + padded, 1])
    }
}
