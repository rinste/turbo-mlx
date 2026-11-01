import Foundation
import MLX
import MLXNN

// MiniMax Music 3's synthesis: the frames' hidden states onto the Flow-VAE's latent timeline (`fusion.py`), the
// one-dimensional flow-matching transformer (`dit.py`) and its guided Euler steps on windows of 200 frames
// (`euler.py`, `_run_flow` in `minimax_music3.py`). Everything runs in the pack's bf16, as the reference does.

/// `ConditionEncoder`: the eight hidden states of each frame mixed by softmax weights, scaled, projected by a
/// convolution, and stretched to the latents' rate by nearest neighbour.
public final class Music3ConditionEncoder: Module {
    @ParameterInfo(key: "layer_weight_logits") var layerWeightLogits: MLXArray
    @ParameterInfo(key: "layer_scale") var layerScale: MLXArray
    @ModuleInfo(key: "proj") var proj: Conv1d
    let config: Music3Config

    init(config: Music3Config) {
        self.config = config
        _layerWeightLogits.wrappedValue = MLXArray.zeros([config.numConditionLayers])
        _layerScale.wrappedValue = MLXArray.ones([1])
        _proj.wrappedValue = Conv1d(inputChannels: config.hiddenSize, outputChannels: config.conditionOutDim, kernelSize: 3, padding: 1)
        super.init()
    }

    /// [B, frames, layers × hidden] → [B, latents, condition].
    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batch, frames) = (hiddenStates.dim(0), hiddenStates.dim(1))
        var hidden = hiddenStates.reshaped(batch, frames, config.numConditionLayers, config.hiddenSize).transposed(0, 2, 3, 1)
        let weights = softmax(layerWeightLogits.asType(.float32), axis: 0).asType(hidden.dtype)
        hidden = (hidden * weights.reshaped(1, -1, 1, 1)).sum(axis: 1)
        hidden = layerScale.asType(hidden.dtype) * hidden
        hidden = proj(hidden.transposed(0, 2, 1)).transposed(0, 2, 1)
        return Self.nearest(hidden, size: config.latentLength(frames: frames)).transposed(0, 2, 1)
    }

    /// `nearest_interpolate_1d` on [N, C, L].
    static func nearest(_ x: MLXArray, size: Int) -> MLXArray {
        let length = x.dim(-1)
        guard size != length else { return x }
        let ratio = Float(Double(length) / Double(size))
        let indices = (MLXArray(Int32(0) ..< Int32(size)).asType(.float32) * ratio).asType(.int32)
        return x.take(clip(indices, min: 0, max: length - 1), axis: 2)
    }
}

public final class Music3FourierEmbedding: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dim: Int) {
        _weight.wrappedValue = MLXArray.zeros([dim / 2, 1])
        super.init()
    }

    public func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let angles = matmul(Float(2 * Double.pi) * timestep.reshaped(-1, 1), weight.T)
        return concatenated([cos(angles), sin(angles)], axis: -1)
    }
}

public final class Music3TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(input: Int, output: Int) {
        _linear1.wrappedValue = Linear(input, output)
        _linear2.wrappedValue = Linear(output, output)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

public final class Music3DiTAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    let heads: Int
    let headDim: Int

    init(dim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        let inner = heads * headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _toOut.wrappedValue = [Linear(inner, dim, bias: false)]
        super.init()
    }

    /// The rotary embedding on the first `cos.dim(-1)` channels of each head, halves rotated; [B, L, H, D].
    static func partialRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let rotaryDim = cos.dim(-1), half = rotaryDim / 2
        let c = cos.expandedDimensions(axis: 1).asType(x.dtype)
        let s = sin.expandedDimensions(axis: 1).asType(x.dtype)
        let rotated = x[.ellipsis, ..<rotaryDim]
        let rotateHalf = concatenated([-rotated[.ellipsis, half...], rotated[.ellipsis, ..<half]], axis: -1)
        return concatenated([rotated * c + rotateHalf * s, x[.ellipsis, rotaryDim...]], axis: -1)
    }

    public func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q = Self.partialRotary(toQ(x).reshaped(batch, length, heads, headDim), cos: cos, sin: sin).transposed(0, 2, 1, 3)
        let k = Self.partialRotary(toK(x).reshaped(batch, length, heads, headDim), cos: cos, sin: sin).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: Float(1.0 / Double(headDim).squareRoot()), mask: .none)
        return toOut[0](y.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

public final class Music3DiTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: Music3DiTAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ff_in") var ffIn: Linear
    @ModuleInfo(key: "ff_out") var ffOut: Linear

    init(dim: Int, heads: Int, headDim: Int, ffInner: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = Music3DiTAttention(dim: dim, heads: heads, headDim: headDim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        _ffIn.wrappedValue = Linear(dim, ffInner * 2)
        _ffOut.wrappedValue = Linear(ffInner, dim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x), cos: cos, sin: sin)
        let parts = split(ffIn(norm2(h)), parts: 2, axis: -1)
        return h + ffOut(parts[0] * silu(parts[1]))
    }
}

/// `FlowMatchingTransformer`: the noisy latents, a zero block and the condition stacked on the channels, a residual
/// 1×1 convolution, the time embedding as a token before them, 36 blocks with rotary positions, back out through
/// another residual 1×1 convolution.
public final class Music3FlowTransformer: Module {
    @ModuleInfo(key: "time_proj") var timeProj: Music3FourierEmbedding
    @ModuleInfo(key: "time_embed") var timeEmbed: Music3TimestepEmbedding
    @ModuleInfo(key: "preprocess_conv") var preprocessConv: Conv1d
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [Music3DiTBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "postprocess_conv") var postprocessConv: Conv1d
    let rotaryDim: Int

    init(config: Music3Config) {
        let inner = config.ditNumHeads * config.ditHeadDim
        let concat = 2 * config.ditInChannels + config.conditionOutDim
        rotaryDim = config.ditRotaryDim
        _timeProj.wrappedValue = Music3FourierEmbedding(dim: config.ditFourierDim)
        _timeEmbed.wrappedValue = Music3TimestepEmbedding(input: config.ditFourierDim, output: inner)
        _preprocessConv.wrappedValue = Conv1d(inputChannels: concat, outputChannels: concat, kernelSize: 1, bias: false)
        _projIn.wrappedValue = Linear(concat, inner, bias: false)
        _blocks.wrappedValue = (0 ..< config.ditNumLayers).map { _ in
            Music3DiTBlock(dim: inner, heads: config.ditNumHeads, headDim: config.ditHeadDim, ffInner: config.ditFFInnerDim)
        }
        _projOut.wrappedValue = Linear(inner, config.ditInChannels, bias: false)
        _postprocessConv.wrappedValue = Conv1d(inputChannels: config.ditInChannels, outputChannels: config.ditInChannels, kernelSize: 1, bias: false)
        super.init()
    }

    /// cos and sin [length, rotaryDim] in float32, the frequencies over both halves.
    func rotary(length: Int) -> (MLXArray, MLXArray) {
        let exponents = MLXArray(stride(from: Int32(0), to: Int32(rotaryDim), by: 2).map { $0 }).asType(.float32) / Float(rotaryDim)
        let inverse = MLXArray(Float(1)) / pow(MLXArray(Float(10_000)), exponents)
        let frequencies = outer(MLXArray(Int32(0) ..< Int32(length)).asType(.float32), inverse)
        let both = concatenated([frequencies, frequencies], axis: -1)
        return (cos(both), sin(both))
    }

    /// The velocity [B, channels, L] of `hidden` [B, channels, L] at `timestep` [B] given `condition` [B, L, condition].
    public func callAsFunction(_ hidden: MLXArray, timestep: MLXArray, condition: MLXArray) -> MLXArray {
        var x = concatenated([hidden, zeros(like: hidden), condition.transposed(0, 2, 1)], axis: 1).transposed(0, 2, 1)
        x = preprocessConv(x) + x
        let time = timeEmbed(timeProj(timestep))
        x = concatenated([time.expandedDimensions(axis: 1), projIn(x)], axis: 1)
        let (c, s) = rotary(length: x.dim(1))
        for block in blocks { x = block(x, cos: c, sin: s) }
        x = projOut(x[0..., 1...])
        x = postprocessConv(x) + x
        return x.transposed(0, 2, 1)
    }
}

/// The flow's Euler steps on one window (`denoise_chunk`) and the windows of a song (`_run_flow`).
public struct Music3Flow {
    public let conditionEncoder: Music3ConditionEncoder
    public let transformer: Music3FlowTransformer
    public let config: Music3Config

    public init(conditionEncoder: Music3ConditionEncoder, transformer: Music3FlowTransformer, config: Music3Config) {
        self.conditionEncoder = conditionEncoder
        self.transformer = transformer
        self.config = config
    }

    /// `make_sigma_schedule`: numpy's float32 linspace from 1 to 1/steps, reversed onto 0…1, then 1.
    public static func sigmas(steps: Int) -> [Float] {
        let start = 1.0, stop = 1.0 / Double(steps)
        let raw: [Float] = (0 ..< steps).map { index in
            if steps == 1 { return Float(start) }
            if index == steps - 1 { return Float(stop) }
            return Float(Double(index) * ((stop - start) / Double(steps - 1)) + start)
        }
        return raw.map { 1 - $0 } + [1]
    }

    /// The windows' first frames: one window for up to 200 frames, else one every 100 frames.
    public static func chunkStarts(frames: Int) -> [Int] {
        frames <= Music3Config.chunkFrames ? [0] : Array(stride(from: 0, to: frames - Music3Config.chunkHop, by: Music3Config.chunkHop))
    }

    /// One window's latents [1, channels, L] from `noise`, and its condition with the previous window's over the
    /// overlap. The overlapping latents are pulled toward the previous window's at every step, and restored at the end.
    public func denoise(
        noise: MLXArray, condition: MLXArray, steps: Int, guidance: Float = Music3Config.ditGuidance,
        previous: (latent: MLXArray, condition: MLXArray)?, step: () throws -> Void = {}
    ) rethrows -> (MLXArray, MLXArray) {
        var condition = condition
        var overlap = 0
        if let previous {
            overlap = min(previous.latent.dim(-1), condition.dim(1))
            if overlap > 0 {
                condition = concatenated([previous.condition[0..., ..<overlap], condition[0..., overlap...]], axis: 1)
            }
        }
        let sigmas = Self.sigmas(steps: steps)
        var x = noise
        let noisePrompt = overlap > 0 ? x[.ellipsis, ..<overlap] : nil
        let empty = zeros(like: condition)
        for index in 0 ..< steps {
            let sigma = Double(sigmas[index]), next = Double(sigmas[index + 1])
            if let noisePrompt, let previous {
                let blend = Float(1.0 - (1.0 - 1e-6) * sigma) * noisePrompt + Float(sigma) * previous.latent[.ellipsis, ..<overlap]
                x = concatenated([blend, x[.ellipsis, overlap...]], axis: -1)
            }
            let timestep = MLX.broadcast(MLXArray(Float(sigma)).asType(x.dtype), to: [x.dim(0)])
            let conditional = transformer(x, timestep: timestep, condition: condition)
            let velocity: MLXArray
            if guidance == 1 {
                velocity = conditional
            } else {
                let unconditional = transformer(x, timestep: timestep, condition: empty)
                velocity = unconditional + (conditional - unconditional) * guidance
            }
            x = x + velocity * Float(next - sigma)
            eval(x)
            try step()
        }
        if let previous, overlap > 0 {
            x = concatenated([previous.latent[.ellipsis, ..<overlap], x[.ellipsis, overlap...]], axis: -1)
            eval(x)
        }
        return (x, condition)
    }
}
