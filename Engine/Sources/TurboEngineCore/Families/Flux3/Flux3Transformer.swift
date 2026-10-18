import Foundation
import MLX
import MLXNN

// The FLUX 3 transformer, after mflux's `flux3_transformer/*`: per-mode blocks first (the image,
// the reference pictures and the text each through their own, attending only within themselves),
// then single-stream blocks over [image, references, text] with each token modulated by its
// mode. Every block is the same: one modulated norm feeding attention and the MLP side by side.
// Names follow the reference so the checkpoint loads unchanged.

/// Rotary tables for the 4-axis ids (t, row, column, layer): the angles in double precision on the
/// host, as the reference computes them (`Flux3PosEmbed`), cos and sin [S, sum(axes)/2] in float32.
struct Flux3PosEmbed {
    let theta: Double
    let axesDim: [Int]

    func callAsFunction(_ ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let axes = axesDim.count
        let rows = ids.shape[0]
        let values = ids.asType(.float32).asArray(Float.self)
        let omegas = axesDim.map { dim in
            stride(from: 0, to: dim, by: 2).map { 1 / Foundation.pow(theta, Double($0) / Double(dim)) }
        }
        let half = omegas.reduce(0) { $0 + $1.count }
        var cosValues = [Float](repeating: 0, count: rows * half)
        var sinValues = [Float](repeating: 0, count: rows * half)
        for row in 0 ..< rows {
            var column = row * half
            for axis in 0 ..< axes {
                let position = Double(values[row * axes + axis])
                for omega in omegas[axis] {
                    let angle = position * omega
                    cosValues[column] = Float(Foundation.cos(angle))
                    sinValues[column] = Float(Foundation.sin(angle))
                    column += 1
                }
            }
        }
        return (MLXArray(cosValues, [rows, half]), MLXArray(sinValues, [rows, half]))
    }
}

/// The rotary tables of one generation: each mode's, for the per-mode blocks, and all of them in
/// the single stream's order. The ids stay the same from step to step, so they are built once.
public struct Flux3Rotary {
    let image: (cos: MLXArray, sin: MLXArray)
    let reference: (cos: MLXArray, sin: MLXArray)?
    let text: (cos: MLXArray, sin: MLXArray)
    let joint: (cos: MLXArray, sin: MLXArray)
}

/// Sinusoidal timestep features (256 wide, cos first) through two linears. The reference carries
/// the timestep in bfloat16, so the features are rounded to it before and after.
final class Flux3TimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let inChannels: Int

    init(inChannels: Int, embeddingDim: Int) {
        self.inChannels = inChannels
        _linear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
        _linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
        super.init()
    }

    /// `timestep` [B] in [0, 1], float32.
    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let t = timestep.asType(.bfloat16) * Float(1000)
        let half = inChannels / 2
        let freqs = exp(-Float(Foundation.log(10000.0)) * MLXArray((0 ..< half).map { Float($0) }) / Float(half))
        let args = t.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let features = concatenated([cos(args), sin(args)], axis: -1).asType(t.dtype)
        return linear2(silu(linear1(features)))
    }
}

/// `silu(vec)` through a linear into (shift, scale, gate), each [B, 1, dim].
final class Flux3Modulation: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(dim: Int) {
        _linear.wrappedValue = Linear(dim, dim * 3, bias: false)
        super.init()
    }

    func callAsFunction(_ vec: MLXArray) -> [MLXArray] {
        linear(silu(vec)).expandedDimensions(axis: 1).split(parts: 3, axis: -1)
    }
}

/// One modulation per mode, as the reference's `{mode: Flux3Modulation}` dictionaries.
final class Flux3Modulations: Module {
    @ModuleInfo(key: "image") var image: Flux3Modulation
    @ModuleInfo(key: "image_cond") var imageCond: Flux3Modulation
    @ModuleInfo(key: "txt") var txt: Flux3Modulation

    init(dim: Int) {
        _image.wrappedValue = Flux3Modulation(dim: dim)
        _imageCond.wrappedValue = Flux3Modulation(dim: dim)
        _txt.wrappedValue = Flux3Modulation(dim: dim)
        super.init()
    }
}

/// The input projections of the image's latents and of the reference pictures' tokens.
final class Flux3XEmbedders: Module {
    @ModuleInfo(key: "image") var image: Linear
    @ModuleInfo(key: "image_cond") var imageCond: Linear

    init(inChannels: Int, dim: Int) {
        _image.wrappedValue = Linear(inChannels, dim, bias: false)
        _imageCond.wrappedValue = Linear(inChannels, dim, bias: false)
        super.init()
    }
}

/// q/k/v of one stream, the per-head RMS norm and the rotation both in float32 (`_norm_rope`),
/// attention in the stream's dtype.
final class Flux3Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear

    let heads: Int
    let headDim: Int
    let scale: Float

    init(dim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        scale = Float(1 / Double(headDim).squareRoot())
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _toOut.wrappedValue = Linear(inner, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rotary: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        func heads(_ t: MLXArray) -> MLXArray {
            t.reshaped([batch, length, self.heads, headDim]).transposed(0, 2, 1, 3)
        }
        let query = Self.normRope(heads(toQ(x)), norm: normQ, rotary: rotary)
        let key = Self.normRope(heads(toK(x)), norm: normK, rotary: rotary)
        let value = heads(toV(x))
        let attended = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value, scale: scale, mask: nil)
        return toOut(attended.transposed(0, 2, 1, 3).reshaped([batch, length, self.heads * headDim]))
    }

    static func normRope(_ x: MLXArray, norm: RMSNorm, rotary: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let normed = MLXFast.rmsNorm(x.asType(.float32), weight: norm.weight.asType(.float32), eps: norm.eps)
        return Flux2Rope.apply(normed, cos: rotary.cos, sin: rotary.sin).asType(x.dtype)
    }
}

/// `x + gate · (attention(x̂) + mlp(x̂))` with `x̂ = (1 + scale) · norm(x) + shift`.
final class Flux3Block: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "attn") var attn: Flux3Attention
    @ModuleInfo(key: "ff") var ff: Flux2FeedForward

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Double) {
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = Flux3Attention(dim: dim, heads: heads, headDim: headDim)
        _ff.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
        super.init()
    }

    /// `mod`: shift, scale and gate, [B, 1, dim] or per token [B, S, dim].
    func callAsFunction(_ x: MLXArray, rotary: (cos: MLXArray, sin: MLXArray), mod: [MLXArray]) -> MLXArray {
        let modulated = (1 + mod[1]) * norm(x) + mod[0]
        return x + mod[2] * (attn(modulated, rotary: rotary) + ff(modulated))
    }
}

/// The per-mode blocks of the image and of the reference pictures (the text's are a list of their own).
final class Flux3ModeBlocks: Module {
    @ModuleInfo(key: "image") var image: [Flux3Block]
    @ModuleInfo(key: "image_cond") var imageCond: [Flux3Block]

    init(count: Int, makeBlock: () -> Flux3Block) {
        _image.wrappedValue = (0 ..< count).map { _ in makeBlock() }
        _imageCond.wrappedValue = (0 ..< count).map { _ in makeBlock() }
        super.init()
    }
}

public final class Flux3Transformer: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: Flux3TimestepEmbedder
    @ModuleInfo(key: "target_timestep_embedder") var targetTimestepEmbedder: Flux3TimestepEmbedder?
    @ModuleInfo(key: "early_stream_modulations") var earlyStreamModulations: Flux3Modulations
    @ModuleInfo(key: "single_stream_modulations") var singleStreamModulations: Flux3Modulations
    @ModuleInfo(key: "x_embedders") var xEmbedders: Flux3XEmbedders
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "mode_blocks") var modeBlocks: Flux3ModeBlocks
    @ModuleInfo(key: "txt_mode_blocks") var txtModeBlocks: [Flux3Block]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux3Block]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let config: Flux3Config.Transformer
    let posEmbed: Flux3PosEmbed

    public var flowMap: Bool { targetTimestepEmbedder != nil }

    public init(config: Flux3Config.Transformer) {
        self.config = config
        let inner = config.innerDim
        posEmbed = Flux3PosEmbed(theta: config.ropeTheta, axesDim: config.ropeAxesDim)
        _timestepEmbedder.wrappedValue = Flux3TimestepEmbedder(inChannels: config.timestepChannels, embeddingDim: inner)
        _targetTimestepEmbedder.wrappedValue = config.flowMap
            ? Flux3TimestepEmbedder(inChannels: config.timestepChannels, embeddingDim: inner) : nil
        _earlyStreamModulations.wrappedValue = Flux3Modulations(dim: inner)
        _singleStreamModulations.wrappedValue = Flux3Modulations(dim: inner)
        _xEmbedders.wrappedValue = Flux3XEmbedders(inChannels: config.inChannels, dim: inner)
        _contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, inner, bias: false)
        let makeBlock = {
            Flux3Block(dim: inner, heads: config.numAttentionHeads, headDim: config.attentionHeadDim, mlpRatio: config.mlpRatio)
        }
        _modeBlocks.wrappedValue = Flux3ModeBlocks(count: config.numLayers, makeBlock: makeBlock)
        _txtModeBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in makeBlock() }
        _singleTransformerBlocks.wrappedValue = (0 ..< config.numSingleLayers).map { _ in makeBlock() }
        _normOut.wrappedValue = AdaLayerNormContinuous(embeddingDim: inner, conditioningDim: inner)
        _projOut.wrappedValue = Linear(inner, config.inChannels, bias: false)
        super.init()
    }

    /// The time embeddings a pass modulates with: the image's and the pictures' and text's (parity checks).
    public func timeEmbeddings(timestep: Float, targetTimestep: Float?) -> (image: MLXArray, context: MLXArray) {
        let t = MLXArray([timestep])
        var image = timestepEmbedder(t)
        var context = timestepEmbedder(MLXArray.zeros(like: t))
        if let targetTimestepEmbedder, let targetTimestep {
            let target = targetTimestepEmbedder(MLXArray([targetTimestep]))
            image = image + target
            context = context + target
        }
        return (image, context)
    }

    /// The rotary tables for ids [S, 4] of the image, of the reference pictures (nil: none) and of the text.
    public func rotary(imageIds: MLXArray, referenceIds: MLXArray?, textIds: MLXArray) -> Flux3Rotary {
        let image = posEmbed(imageIds)
        let reference = referenceIds.map { posEmbed($0) }
        let text = posEmbed(textIds)
        let parts = [image] + (reference.map { [$0] } ?? []) + [text]
        let joint = (cos: concatenated(parts.map(\.cos), axis: 0), sin: concatenated(parts.map(\.sin), axis: 0))
        return Flux3Rotary(image: image, reference: reference, text: text, joint: joint)
    }

    /// One pass: the velocity [B, S, 128] for the image's latents [B, S, 128] at `timestep` (in
    /// [0, 1]), towards `targetTimestep` for a flow map (nil otherwise). `references` are the
    /// pictures' tokens [B, R, 128] (nil: text to image), `prompt` [B, T, joint dim].
    public func callAsFunction(
        latents: MLXArray, prompt: MLXArray, references: MLXArray?, timestep: Float, targetTimestep: Float?, rotary: Flux3Rotary
    ) -> MLXArray {
        let batch = latents.shape[0]
        let t = MLXArray(Array(repeating: timestep, count: batch))
        // Image tokens see the current time, the pictures' and the text's t = 0; a flow map adds the
        // target time to every mode.
        var vecImage = timestepEmbedder(t)
        var vecContext = timestepEmbedder(MLXArray.zeros(like: t))
        if let targetTimestepEmbedder, let targetTimestep {
            let target = targetTimestepEmbedder(MLXArray(Array(repeating: targetTimestep, count: batch)))
            vecImage = vecImage + target
            vecContext = vecContext + target
        }
        let early = (image: earlyStreamModulations.image(vecImage), cond: earlyStreamModulations.imageCond(vecContext),
                     txt: earlyStreamModulations.txt(vecContext))
        let single = (image: singleStreamModulations.image(vecImage), cond: singleStreamModulations.imageCond(vecContext),
                      txt: singleStreamModulations.txt(vecContext))

        var image = xEmbedders.image(latents.asType(modelPrecision))
        var cond = references.flatMap { $0.shape[1] > 0 ? xEmbedders.imageCond($0.asType(modelPrecision)) : nil }
        var text = contextEmbedder(prompt.asType(modelPrecision))

        for d in 0 ..< txtModeBlocks.count {
            image = modeBlocks.image[d](image, rotary: rotary.image, mod: early.image)
            if let current = cond, let referenceRotary = rotary.reference {
                cond = modeBlocks.imageCond[d](current, rotary: referenceRotary, mod: early.cond)
            }
            text = txtModeBlocks[d](text, rotary: rotary.text, mod: early.txt)
        }

        let streams = [(image, single.image)] + (cond.map { [($0, single.cond)] } ?? []) + [(text, single.txt)]
        var x = concatenated(streams.map(\.0), axis: 1)
        let dim = x.shape[2]
        let mod = (0 ..< 3).map { i in
            concatenated(streams.map { stream, modulation in
                broadcast(modulation[i], to: [batch, stream.shape[1], dim])
            }, axis: 1)
        }
        for block in singleTransformerBlocks {
            x = block(x, rotary: rotary.joint, mod: mod)
        }
        let imageCount = latents.shape[1]
        return projOut(normOut(x[0..., 0 ..< imageCount], temb: vecImage))
    }
}
