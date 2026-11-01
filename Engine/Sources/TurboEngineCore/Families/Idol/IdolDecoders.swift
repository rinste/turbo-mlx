import Foundation
import MLX
import MLXNN

// Idol's two decoders, after idol-sdk's `models/decoder.py` (speridlabs, Apache-2.0). The octree decoder places the
// anchors: from the whole cube down eight levels, it predicts how each cell's points split among its eight children,
// read off the latent by cross attention. The Gaussian decoder turns each anchor into 32 Gaussians, its own tokens
// attending to each other and to the latent. Both run their blocks in float16 (8-bit linears in the MLX pack), their
// embeddings, norms and heads in float32, as the reference.

/// `PcdAbsolutePositionEmbedderv2`: each coordinate's sines then cosines at 170 frequencies 2^0…2^10 (times π), the
/// three coordinates one after the other, zeros up to the width. `x` [N, 3] in [0, 1].
func idolPointEmbedding(_ x: MLXArray, channels: Int, maxRes: Float = 10) -> MLXArray {
    let count = channels / 3 / 2
    // torch.linspace(0, max_res, count) in float32: its first half from the start, its second from the end.
    let step = maxRes / Float(count - 1)
    let logs = (0 ..< count).map { i -> Float in i < count / 2 ? Float(i) * step : maxRes - Float(count - 1 - i) * step }
    let frequencies = pow(MLXArray(Float(2)), MLXArray(logs))
    let angles = x.expandedDimensions(axis: -1) * frequencies * Float.pi
    var embedding = concatenated([MLX.sin(angles), MLX.cos(angles)], axis: -1).reshaped([x.dim(0), -1])
    if embedding.dim(-1) < channels {
        embedding = concatenated([embedding, MLXArray.zeros([x.dim(0), channels - embedding.dim(-1)], dtype: embedding.dtype)], axis: -1)
    }
    return embedding
}

/// `MultiHeadAttention`: self attention (`to_qkv`) or cross attention to a context (`to_q`, `to_kv`), queries and keys
/// RMS-normalized per head, no rotation.
final class IdolDecoderAttention: Module {
    @ModuleInfo(key: "to_qkv") var toQKV: Linear?
    @ModuleInfo(key: "to_q") var toQ: Linear?
    @ModuleInfo(key: "to_kv") var toKV: Linear?
    @ModuleInfo(key: "q_rms_norm") var qNorm: IdolHeadNorm
    @ModuleInfo(key: "k_rms_norm") var kNorm: IdolHeadNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int

    init(dim: Int, heads: Int, contextDim: Int? = nil) {
        self.heads = heads
        if let contextDim {
            _toQ.wrappedValue = Linear(dim, dim)
            _toKV.wrappedValue = Linear(contextDim, 2 * dim)
        } else {
            _toQKV.wrappedValue = Linear(dim, 3 * dim)
        }
        _qNorm.wrappedValue = IdolHeadNorm(heads: heads, headDim: dim / heads)
        _kNorm.wrappedValue = IdolHeadNorm(heads: heads, headDim: dim / heads)
        _toOut.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = dim / heads
        let query: MLXArray, key: MLXArray, value: MLXArray
        if let context, let toQ, let toKV {
            query = qNorm(toQ(x).reshaped([batch, length, heads, headDim]))
            let kv = toKV(context).reshaped([batch, context.dim(1), 2, heads, headDim])
            key = kNorm(kv[0..., 0..., 0])
            value = kv[0..., 0..., 1]
        } else {
            let qkv = toQKV!(x).reshaped([batch, length, 3, heads, headDim])
            query = qNorm(qkv[0..., 0..., 0])
            key = kNorm(qkv[0..., 0..., 1])
            value = qkv[0..., 0..., 2]
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: query.transposed(0, 2, 1, 3), keys: key.transposed(0, 2, 1, 3), values: value.transposed(0, 2, 1, 3),
            scale: 1 / Float(headDim).squareRoot(), mask: nil
        )
        return toOut(attended.transposed(0, 2, 1, 3).reshaped([batch, length, dim]))
    }
}

/// `ModulatedTransformerCrossOnlyBlock`: cross attention to the latent and a feed-forward, both modulated by the
/// octree level's adaLN (no table of its own).
final class IdolOctreeBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: IdolDecoderAttention
    @ModuleInfo(key: "mlp") var mlp: IdolFeedForward

    init(dim: Int, contextDim: Int, heads: Int, mlpDim: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _crossAttn.wrappedValue = IdolDecoderAttention(dim: dim, heads: heads, contextDim: contextDim)
        _mlp.wrappedValue = IdolFeedForward(input: dim, hidden: mlpDim, output: dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, modulation: MLXArray, context: MLXArray) -> MLXArray {
        let mod = modulation.expandedDimensions(axis: 1).split(parts: 6, axis: -1)
        var h = idolNorm32(norm1, x) * (1 + mod[1]) + mod[0]
        h = crossAttn(h, context: context)
        let x = x + h * mod[2]
        h = mlp(idolNorm32(norm2, x) * (1 + mod[4]) + mod[3])
        return x + h * mod[5]
    }
}

/// `TransformerCrossBlock`: self attention, cross attention to the latent (after an affine norm), a feed-forward.
final class IdolGaussianBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: IdolDecoderAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: IdolDecoderAttention
    @ModuleInfo(key: "mlp") var mlp: IdolFeedForward

    init(dim: Int, contextDim: Int, heads: Int, mlpDim: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _norm3.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _selfAttn.wrappedValue = IdolDecoderAttention(dim: dim, heads: heads)
        _crossAttn.wrappedValue = IdolDecoderAttention(dim: dim, heads: heads, contextDim: contextDim)
        _mlp.wrappedValue = IdolFeedForward(input: dim, hidden: mlpDim, output: dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var x = x + selfAttn(idolNorm32(norm1, x))
        x = x + crossAttn(idolNorm32(norm2, x), context: context)
        return x + mlp(idolNorm32(norm3, x))
    }
}

/// `SinusoidalEmbedder(max_period=1024, two_pi=True)` for the octree's level: 256 features (cos first) of 2π·res.
final class IdolLevelEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(dim: Int) {
        _linear1.wrappedValue = Linear(256, dim)
        _linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ level: MLXArray) -> MLXArray {
        let half = 128
        let frequencies = exp(-Float(log(1024.0)) * MLXArray((0 ..< half).map { Float($0) }) / Float(half))
        let args = level.asType(.float32).expandedDimensions(axis: 1) * frequencies.expandedDimensions(axis: 0) * 2 * Float.pi
        return linear2(silu(linear1(concatenated([MLX.cos(args), MLX.sin(args)], axis: -1))))
    }
}

/// `OctreeProbabilityFixedlenDecoder`.
public final class IdolOctreeDecoder: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "l_embedder") var levelEmbedder: IdolLevelEmbedder
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear
    @ModuleInfo(key: "blocks") var blocks: [IdolOctreeBlock]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let dim: Int

    public init(config: IdolConfig.Decoder) {
        dim = config.dim
        _inProj.wrappedValue = Linear(3, dim)
        _inputLayer.wrappedValue = Linear(dim, dim)
        _levelEmbedder.wrappedValue = IdolLevelEmbedder(dim: dim)
        _adaLNModulation.wrappedValue = Linear(dim, 6 * dim)
        _blocks.wrappedValue = (0 ..< config.blocks).map { _ in
            IdolOctreeBlock(dim: config.dim, contextDim: config.contextChannels, heads: config.heads, mlpDim: config.mlpDim)
        }
        _outProj.wrappedValue = Linear(dim, 8)
        super.init()
    }

    /// Cell centres [1, L, 3] in [0, 1] at resolution `resolution` (2^level), the latent [1, 8192, 16] → the logits of
    /// how each cell's points split among its eight children [1, L, 8], float32.
    public func callAsFunction(_ centres: MLXArray, resolution: Int, latent: MLXArray) -> MLXArray {
        let (batch, length) = (centres.dim(0), centres.dim(1))
        var h = inProj(centres) + idolPointEmbedding(centres.reshaped([-1, 3]), channels: dim).reshaped([batch, length, -1])
        h = inputLayer(h)
        let modulation = adaLNModulation(silu(levelEmbedder(MLXArray([Float(resolution)])))).asType(idolStreamType)
        var out = h.asType(idolStreamType)
        let context = latent.asType(idolStreamType)
        for block in blocks { out = block(out, modulation: modulation, context: context) }
        return outProj(MLXFast.layerNorm(out.asType(.float32), weight: nil, bias: nil, eps: 1e-5))
    }

    /// `OctreeProbabilityFixedlenDecoder.sample`: `count` anchor points [1, count, 3] in [0, 1], each cell's points
    /// split among its children by systematic resampling from its predicted probabilities, down to `levels`, then
    /// placed uniformly at random inside their leaf cell. The draws are this engine's (`random`); only their law is
    /// the reference's.
    public func sample(latent: MLXArray, count: Int, levels: Int, random: inout IdolRandom, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        var cells: [(Int, Int, Int)] = [(0, 0, 0)]
        var counts = [count]
        for level in 1 ... levels {
            if isCancelled() { throw CancellationError() }
            let parentResolution = Float(1 << (level - 1))
            var centres = [Float]()
            centres.reserveCapacity(cells.count * 3)
            for (x, y, z) in cells {
                centres.append((Float(x) + 0.5) / parentResolution)
                centres.append((Float(y) + 0.5) / parentResolution)
                centres.append((Float(z) + 0.5) / parentResolution)
            }
            let logits = self(MLXArray(centres).reshaped([1, cells.count, 3]), resolution: 1 << level, latent: latent)
            let probabilities = softmax(logits, axis: -1).asType(.float32).asArray(Float.self)
            var nextCells: [(Int, Int, Int)] = []
            var nextCounts: [Int] = []
            for (index, cell) in cells.enumerated() {
                let split = Self.systematic(Array(probabilities[(index * 8) ..< (index * 8 + 8)]), count: counts[index], random: &random)
                // Children in the reference's order: x fastest, then y, then z.
                for child in 0 ..< 8 where split[child] > 0 {
                    nextCells.append((cell.0 * 2 + (child & 1), cell.1 * 2 + ((child >> 1) & 1), cell.2 * 2 + ((child >> 2) & 1)))
                    nextCounts.append(split[child])
                }
            }
            (cells, counts) = (nextCells, nextCounts)
        }
        let resolution = Float(1 << levels)
        var points = [Float]()
        points.reserveCapacity(count * 3)
        for (cell, n) in zip(cells, counts) {
            for _ in 0 ..< n {
                points.append((Float(cell.0) + random.uniform()) / resolution)
                points.append((Float(cell.1) + random.uniform()) / resolution)
                points.append((Float(cell.2) + random.uniform()) / resolution)
            }
        }
        return MLXArray(points).reshaped([1, count, 3])
    }

    /// `systematic_sample` for one cell: `count` points among the eight children, one random offset and evenly spaced
    /// positions on the cumulative probabilities (normalized when they add up to more than one, as the reference's
    /// `clamp_min(1)` does).
    static func systematic(_ probabilities: [Float], count: Int, random: inout IdolRandom) -> [Int] {
        var split = [Int](repeating: 0, count: probabilities.count)
        guard count > 0 else { return split }
        var p = probabilities.map { Swift.max($0, 0) }
        let total = p.reduce(0, +)
        if total == 0 {
            p = [Float](repeating: 1 / Float(p.count), count: p.count)
        } else {
            let divisor = Swift.max(total, 1)
            p = p.map { $0 / divisor }
        }
        var cdf = [Float](repeating: 0, count: p.count)
        var running: Float = 0
        for (i, value) in p.enumerated() {
            running += value
            cdf[i] = Swift.min(running, 1)
        }
        let n = Float(count)
        let offset = random.uniform() / n
        for k in 0 ..< count {
            let u = Swift.min(offset + Float(k) / n, 1)
            // searchsorted (left): the first bin whose cumulative probability reaches u.
            var bin = cdf.firstIndex { $0 >= u } ?? (cdf.count - 1)
            bin = Swift.min(bin, cdf.count - 1)
            split[bin] += 1
        }
        return split
    }
}

/// The Gaussians of an object, activated as idol-sdk's `Gaussian` does: positions in the model's z-up frame within
/// [-0.5, 0.5]³, degree-0 colour coefficients, scales, unit rotations (w, x, y, z), opacities.
public struct IdolGaussians {
    public var xyz: [Float]
    public var colour: [Float]
    public var scale: [Float]
    public var rotation: [Float]
    public var opacity: [Float]
    public var count: Int { opacity.count }
}

/// `GaussianFixedlenDecoder`.
public final class IdolGaussianDecoder: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "blocks") var blocks: [IdolGaussianBlock]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    /// The fixed offset pattern of an anchor's Gaussians [32, 3], and the base of their spread.
    @ParameterInfo(key: "points_offset_perturbation") var offsetPerturbation: MLXArray
    @ParameterInfo(key: "base_offset_scale") var baseOffsetScale: MLXArray
    let config: IdolConfig.GaussianDecoder

    public init(config: IdolConfig.GaussianDecoder) {
        self.config = config
        let dim = config.dim
        _inProj.wrappedValue = Linear(3, dim)
        _inputLayer.wrappedValue = Linear(dim, dim)
        _blocks.wrappedValue = (0 ..< config.blocks).map { _ in
            IdolGaussianBlock(dim: dim, contextDim: config.contextChannels, heads: config.heads, mlpDim: config.mlpDim)
        }
        _outProj.wrappedValue = Linear(dim, config.perAnchor * 15)
        _offsetPerturbation.wrappedValue = MLXArray.zeros([config.perAnchor, 3])
        _baseOffsetScale.wrappedValue = MLXArray(Float(0))
        super.init()
    }

    /// Anchors [1, L, 3], the latent [1, 8192, 16] → each anchor's raw outputs [1, L, 32 × 15], float32.
    public func callAsFunction(_ points: MLXArray, latent: MLXArray) -> MLXArray {
        let (batch, length) = (points.dim(0), points.dim(1))
        var h = inProj(points) + idolPointEmbedding(points.reshaped([-1, 3]), channels: config.dim).reshaped([batch, length, -1])
        h = inputLayer(h)
        var out = h.asType(idolStreamType)
        let context = latent.asType(idolStreamType)
        for block in blocks { out = block(out, context: context) }
        return outProj(MLXFast.layerNorm(out.asType(.float32), weight: nil, bias: nil, eps: 1e-5))
    }

    /// `to_representation` and the `Gaussian` activations, for one object: [L, 32 × 15] → L × 32 Gaussians.
    public func activate(points: MLXArray, raw: MLXArray) -> (xyz: MLXArray, colour: MLXArray, scale: MLXArray, rotation: MLXArray, opacity: MLXArray) {
        let n = config.perAnchor
        let length = raw.dim(1)
        // The layout of an anchor's outputs: offsets (3), colour (3), scales (3), rotation (4), opacity (1), spread (1),
        // each for the anchor's 32 Gaussians in turn.
        var start = 0
        func slice(_ width: Int) -> MLXArray {
            defer { start += n * width }
            return raw[0, 0..., start ..< (start + n * width)].reshaped([length, n, width])
        }
        let lr = config.learningRates
        let rawOffset = slice(3), rawColour = slice(3), rawScale = slice(3), rawRotation = slice(4), rawOpacity = slice(1), rawSpread = slice(1)
        let spread = softplus(rawSpread + baseOffsetScale)
        var offset = rawOffset * lr.xyz + offsetPerturbation
        offset = tanh(offset) * 0.5 * config.perturbSize * spread
        let xyz = (offset + points[0].expandedDimensions(axis: 1)).reshaped([-1, 3]) - 0.5
        let colour = (rawColour * lr.colour).reshaped([-1, 3])
        let scaleOffset = config.scalingBias + Float(log(-expm1(-Double(config.scalingBias))))
        var scale = softplus((rawScale * lr.scaling).reshaped([-1, 3]) + scaleOffset)
        scale = sqrt(scale * scale + config.minimumKernelSize * config.minimumKernelSize)
        var rotation = (rawRotation * lr.rotation).reshaped([-1, 4]) + MLXArray([Float(1), 0, 0, 0])
        rotation = rotation / maximum(sqrt((rotation * rotation).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        let opacityOffset = Float(log(Double(config.opacityBias) / (1 - Double(config.opacityBias))))
        let opacity = sigmoid((rawOpacity * lr.opacity).reshaped([-1]) + opacityOffset)
        return (xyz, colour, scale, rotation, opacity)
    }

    /// The anchors' Gaussians, decoded and read back.
    public func decode(points: MLXArray, latent: MLXArray) -> IdolGaussians {
        let g = activate(points: points, raw: self(points, latent: latent))
        eval(g.xyz, g.colour, g.scale, g.rotation, g.opacity)
        return IdolGaussians(xyz: g.xyz.asArray(Float.self), colour: g.colour.asArray(Float.self), scale: g.scale.asArray(Float.self),
                             rotation: g.rotation.asArray(Float.self), opacity: g.opacity.asArray(Float.self))
    }
}

/// The engine's own draws for the octree (its systematic offsets, the points' places in their cells): splitmix64,
/// seeded with the request's seed.
public struct IdolRandom {
    private var state: UInt64

    public init(seed: Int) { state = UInt64(bitPattern: Int64(seed)) &+ 0x9E37_79B9_7F4A_7C15 }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1), 24 bits.
    public mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
}
