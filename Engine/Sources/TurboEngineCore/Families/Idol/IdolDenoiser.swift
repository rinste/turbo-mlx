import Foundation
import MLX
import MLXNN

// Idol's denoiser, after idol-sdk's `models/dit.py` and `models/layers.py` (speridlabs, Apache-2.0): a rectified-flow
// transformer over the object's latent (8,192 tokens of 16) and the input camera (one token of 5), conditioned on the
// picture's tokens (DINOv3's and the FLUX.2 VAE's, 4,101 each, summed after their own projections). Each stream is
// refined on its own (the latent by two blocks modulated by the time, the picture by two plain ones, the camera by an
// MLP); then [latent, picture, camera] go through the joint blocks. Positions are learned per token: before each block
// a RePo layer reads the tokens' features and turns them into rotary angles, three axes per head.
//
// Precision as the reference runs it: the blocks' linears (8 bits in the MLX pack) and the residual stream in float16;
// norms, embeddings, the RePo layers, the rotation, the heads' RMS norm and the output layers in float32. The rotation
// comes before the q/k norm, on adjacent (even, odd) pairs. Names follow the checkpoint, the `Sequential`s' numbered
// layers renamed on loading (`IdolDenoiser.load`), so the pack loads as it is.

/// The residual stream's type: idol-sdk's `self.dtype`.
let idolStreamType = DType.float16

/// `LayerNorm32`: the norm in float32, given back in the input's type.
func idolNorm32(_ norm: LayerNorm, _ x: MLXArray) -> MLXArray {
    norm(x.asType(.float32)).asType(x.dtype)
}

/// `apply_rotary_emb` with real (cos, sin) tables: x [B, L, H, D] rotated pair by pair in float32, given back in its
/// type. `cos`, `sin` [B, L, H, D/2].
func idolRotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let pairs = x.asType(.float32).reshaped(Array(x.shape.dropLast()) + [x.dim(-1) / 2, 2])
    let even = pairs[.ellipsis, 0]
    let odd = pairs[.ellipsis, 1]
    return stacked([even * cos - odd * sin, even * sin + odd * cos], axis: -1).reshaped(x.shape).asType(x.dtype)
}

/// `MultiHeadRMSNorm`: each head's vector over its length, times a gain per head and channel and √D, in float32.
final class IdolHeadNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float

    init(heads: Int, headDim: Int) {
        _gamma.wrappedValue = MLXArray.ones([heads, headDim])
        scale = Float(headDim).squareRoot()
        super.init()
    }

    /// x [B, L, H, D].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        // F.normalize: x / max(‖x‖, 1e-12).
        let length = maximum(sqrt((x32 * x32).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return (x32 / length * gamma * scale).asType(x.dtype)
    }
}

/// `RePo3DRotaryEmbedding`: each token's features give a 3D offset per head, its three axes turned into angles at
/// learned frequencies (`clamp_mul`'s forward value, x·tanh(f) + x·(f − tanh f), times π).
final class IdolRePo: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "gate_map") var gateMap: Linear
    @ModuleInfo(key: "content_map") var contentMap: Linear
    @ModuleInfo(key: "final_map") var finalMap: Linear
    @ParameterInfo(key: "freqs_0") var freqs0: MLXArray
    @ParameterInfo(key: "freqs_1") var freqs1: MLXArray
    @ParameterInfo(key: "freqs_2") var freqs2: MLXArray
    let heads: Int

    init(dim: Int, heads: Int, headDim: Int, hiddenRatio: Double = 0.125) {
        self.heads = heads
        let hidden = Int(Double(dim) * hiddenRatio)
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
        _gateMap.wrappedValue = Linear(dim, hidden, bias: false)
        _contentMap.wrappedValue = Linear(dim, hidden, bias: false)
        _finalMap.wrappedValue = Linear(hidden, 3 * heads, bias: false)
        // The head's channels in three parts, x and y even-sized: 20, 20 and 24 for 64.
        let first = 2 * (headDim / 6)
        let sizes = [first, first, headDim - 2 * first]
        _freqs0.wrappedValue = MLXArray.zeros([sizes[0] / 2])
        _freqs1.wrappedValue = MLXArray.zeros([sizes[1] / 2])
        _freqs2.wrappedValue = MLXArray.zeros([sizes[2] / 2])
        super.init()
    }

    /// The rotation of every token and head, (cos, sin) [B, L, H, D/2] in float32.
    func callAsFunction(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let h = norm(hidden.asType(.float32))
        let out = finalMap(silu(gateMap(h)) * contentMap(h))
        let offsets = out.reshaped([out.dim(0), out.dim(1), heads, 3])
        func angles(_ axis: Int, _ freqs: MLXArray) -> MLXArray {
            let x = offsets[.ellipsis, axis].expandedDimensions(axis: -1)
            let squashed = tanh(freqs)
            return (x * squashed + x * (freqs - squashed)) * Float.pi
        }
        let angle = concatenated([angles(0, freqs0), angles(1, freqs1), angles(2, freqs2)], axis: -1)
        return (MLX.cos(angle), MLX.sin(angle))
    }
}

/// `RopeMultiHeadAttention`: one fused projection with its bias, the rotation then the RMS norm on queries and keys.
final class IdolAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "q_norm") var qNorm: IdolHeadNorm
    @ModuleInfo(key: "k_norm") var kNorm: IdolHeadNorm
    @ModuleInfo(key: "out") var out: Linear
    let heads: Int
    let headDim: Int

    init(dim: Int, heads: Int, headDim: Int) {
        self.heads = heads
        self.headDim = headDim
        _qkv.wrappedValue = Linear(dim, 3 * dim)
        _qNorm.wrappedValue = IdolHeadNorm(heads: heads, headDim: headDim)
        _kNorm.wrappedValue = IdolHeadNorm(heads: heads, headDim: headDim)
        _out.wrappedValue = Linear(dim, dim)
        super.init()
    }

    /// `mask`: an additive bias per key, [1, 1, 1, L] in the stream's type (`IdolDenoiser.collapsed`).
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let parts = qkv(x).reshaped([batch, length, 3, heads, headDim])
        let query = qNorm(idolRotate(parts[0..., 0..., 0], cos: cos, sin: sin))
        let key = kNorm(idolRotate(parts[0..., 0..., 1], cos: cos, sin: sin))
        let value = parts[0..., 0..., 2]
        let attended = MLXFast.scaledDotProductAttention(
            queries: query.transposed(0, 2, 1, 3), keys: key.transposed(0, 2, 1, 3), values: value.transposed(0, 2, 1, 3),
            scale: 1 / Float(headDim).squareRoot(), mask: mask
        )
        return out(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }
}

/// `FeedForwardNet` (and the camera's `MLP`): a linear, GELU's tanh form, a linear.
final class IdolFeedForward: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(input: Int, hidden: Int, output: Int) {
        _linear1.wrappedValue = Linear(input, hidden)
        _linear2.wrappedValue = Linear(hidden, output)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(geluApproximate(linear1(x))) }
}

/// `UnifiedTransformerBlock` with modulation (the latent's refiner and the joint blocks): non-affine norms, the shared
/// adaLN of the time plus this block's own shift table.
final class IdolBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: IdolAttention
    @ModuleInfo(key: "mlp") var mlp: IdolFeedForward
    @ParameterInfo(key: "shift_table") var shiftTable: MLXArray

    init(dim: Int, heads: Int, headDim: Int, mlpDim: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = IdolAttention(dim: dim, heads: heads, headDim: headDim)
        _mlp.wrappedValue = IdolFeedForward(input: dim, hidden: mlpDim, output: dim)
        _shiftTable.wrappedValue = MLXArray.zeros([1, 6 * dim])
        super.init()
    }

    /// `modulation` [B, 6·dim] in the stream's type.
    func callAsFunction(_ x: MLXArray, modulation: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let mod = (modulation + shiftTable.asType(modulation.dtype)).expandedDimensions(axis: 1).split(parts: 6, axis: -1)
        var h = idolNorm32(norm1, x) * (1 + mod[1]) + mod[0]
        h = attn(h, cos: cos, sin: sin, mask: mask)
        let x = x + h * mod[2]
        h = mlp(idolNorm32(norm2, x) * (1 + mod[4]) + mod[3])
        return x + h * mod[5]
    }
}

/// `UnifiedTransformerBlock` without modulation (the picture's refiner): affine norms, plain residuals.
final class IdolRefinerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: IdolAttention
    @ModuleInfo(key: "mlp") var mlp: IdolFeedForward

    init(dim: Int, heads: Int, headDim: Int, mlpDim: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _attn.wrappedValue = IdolAttention(dim: dim, heads: heads, headDim: headDim)
        _mlp.wrappedValue = IdolFeedForward(input: dim, hidden: mlpDim, output: dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let x = x + attn(idolNorm32(norm1, x), cos: cos, sin: sin)
        return x + mlp(idolNorm32(norm2, x))
    }
}

/// `SinusoidalEmbedder` for the time: 256 sinusoidal features (cos first) through a linear, SiLU and a linear.
final class IdolTimeEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    static let frequencies = 256

    init(dim: Int) {
        _linear1.wrappedValue = Linear(Self.frequencies, dim)
        _linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    /// `t` [B] in [0, 1000], float32.
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        linear2(silu(linear1(Flux2TimestepGuidanceEmbeddings.features(t.asType(.float32), dim: Self.frequencies))))
    }
}

public final class IdolDenoiser: Module {
    @ModuleInfo(key: "t_embedder") var tEmbedder: IdolTimeEmbedder
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Linear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "cond_embedder") var condEmbedder: Linear
    @ModuleInfo(key: "cond_embedder2") var condEmbedder2: Linear
    /// The latent tokens' positions in [0, 1]³: a fixed scrambled Sobol sequence stored with the weights.
    @ParameterInfo(key: "pos_pe") var positions: MLXArray
    @ModuleInfo(key: "noise_repo_layers") var noiseRepoLayers: [IdolRePo]
    @ModuleInfo(key: "context_repo_layers") var contextRepoLayers: [IdolRePo]
    @ModuleInfo(key: "repo_layers") var repoLayers: [IdolRePo]
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [IdolBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [IdolRefinerBlock]
    @ModuleInfo(key: "cam_refiner") var camRefiner: IdolFeedForward
    @ModuleInfo(key: "blocks") var blocks: [IdolBlock]
    @ParameterInfo(key: "shift_table") var shiftTable: MLXArray
    @ModuleInfo(key: "out_layer") var outLayer: Linear
    @ModuleInfo(key: "cam_out_layer") var camOutLayer: Linear

    public let config: IdolConfig.Denoiser
    /// The latent tokens' sinusoidal embedding, computed once from `positions` (not a parameter).
    private let cache = Cache()
    private final class Cache { var positionEmbedding: MLXArray? }

    public init(config: IdolConfig.Denoiser) {
        self.config = config
        let (dim, heads, headDim, mlpDim) = (config.dim, config.heads, config.headDim, config.mlpDim)
        _tEmbedder.wrappedValue = IdolTimeEmbedder(dim: dim)
        _adaLNModulation.wrappedValue = Linear(dim, 6 * dim)
        _inputLayer.wrappedValue = Linear(config.latentChannels, dim)
        _condEmbedder.wrappedValue = Linear(config.condChannels, dim)
        _condEmbedder2.wrappedValue = Linear(config.cond2Channels, dim)
        _positions.wrappedValue = MLXArray.zeros([1, config.tokens, 3])
        func repo(_ count: Int) -> [IdolRePo] { (0 ..< count).map { _ in IdolRePo(dim: dim, heads: heads, headDim: headDim) } }
        _noiseRepoLayers.wrappedValue = repo(config.refinerBlocks)
        _contextRepoLayers.wrappedValue = repo(config.refinerBlocks)
        _repoLayers.wrappedValue = repo(config.blocks)
        _noiseRefiner.wrappedValue = (0 ..< config.refinerBlocks).map { _ in IdolBlock(dim: dim, heads: heads, headDim: headDim, mlpDim: mlpDim) }
        _contextRefiner.wrappedValue = (0 ..< config.refinerBlocks).map { _ in IdolRefinerBlock(dim: dim, heads: heads, headDim: headDim, mlpDim: mlpDim) }
        // `MLP(cam_channels, model_channels, model_channels, mlp_layer_num=num_refiner_blocks)`: two layers.
        _camRefiner.wrappedValue = IdolFeedForward(input: config.cameraChannels, hidden: dim, output: dim)
        _blocks.wrappedValue = (0 ..< config.blocks).map { _ in IdolBlock(dim: dim, heads: heads, headDim: headDim, mlpDim: mlpDim) }
        _shiftTable.wrappedValue = MLXArray.zeros([1, 2, dim])
        _outLayer.wrappedValue = Linear(dim, config.latentChannels)
        _camOutLayer.wrappedValue = Linear(dim, config.cameraChannels)
        super.init()
    }

    /// The denoiser of a checkpoint folder (the release's or the MLX pack's): `config.json` and `denoiser.safetensors`,
    /// its 8-bit layers loaded as they are stored.
    public static func load(folder: URL) throws -> IdolDenoiser {
        let config = try IdolConfig.read(folder: folder)
        let model = IdolDenoiser(config: config.denoiser)
        var tensors: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: folder.appending(path: "denoiser.safetensors")) {
            tensors[renamed(key)] = value
        }
        try WeightLoading.apply(tensors, to: model)
        eval(model)
        return model
    }

    /// The `nn.Sequential`s' numbered layers by the names the modules give them: `t_embedder.mlp.0` and `.2` (with a
    /// SiLU between), `cam_refiner.mlp.0` and `.2`, a block's `mlp.mlp.0` and `.2` (GELU between), and
    /// `adaLN_modulation.1` (after a SiLU).
    static func renamed(_ key: String) -> String {
        key.replacingOccurrences(of: ".mlp.0.", with: ".linear_1.")
            .replacingOccurrences(of: ".mlp.2.", with: ".linear_2.")
            .replacingOccurrences(of: "adaLN_modulation.1.", with: "adaLN_modulation.")
    }

    /// `PcdAbsolutePositionEmbedder`: each coordinate's sines then cosines at 170 frequencies (2⁰…2¹⁵, then 154 more
    /// spread over the same range), the three coordinates one after the other, zeros up to the width.
    func positionEmbedding() -> MLXArray {
        if let cached = cache.positionEmbedding { return cached }
        let dim = config.dim
        let count = dim / 3 / 2
        let maxRes = 16
        let spread = Swift.max(0, count - maxRes)
        let exponents = ((0 ..< maxRes).map { Float($0) } + (0 ..< spread).map { Float($0) / Float(spread) * Float(maxRes) }).prefix(count)
        let frequencies = pow(MLXArray(Float(2)), MLXArray(Array(exponents)))
        let x = positions.reshaped([-1])
        let angles = x.expandedDimensions(axis: 1) * frequencies.expandedDimensions(axis: 0) * 2 * Float.pi
        var embedding = concatenated([MLX.sin(angles), MLX.cos(angles)], axis: -1).reshaped([1, config.tokens, -1])
        if embedding.dim(-1) < dim {
            embedding = concatenated([embedding, MLXArray.zeros([1, config.tokens, dim - embedding.dim(-1)])], axis: -1)
        }
        eval(embedding)
        cache.positionEmbedding = embedding
        return embedding
    }

    /// What one step computes once for both guidance passes: the latent refined, the camera's token, the time's
    /// modulation and embedding.
    public struct Stream {
        public var latent: MLXArray
        public var camera: MLXArray
        /// `adaLN_modulation(t_emb)` [B, 6·dim] in the stream's type, and `t_emb` [B, dim] in float32.
        public var modulation: MLXArray
        public var timeEmbedding: MLXArray
    }

    /// The picture's tokens refined (DINOv3's [B, 4101, 1280], the VAE's [B, 4101, 128]), [B, 4101, dim] in the
    /// stream's type: once per guidance branch, the unconditional one from zeros.
    public func context(feature1: MLXArray, feature2: MLXArray) -> MLXArray {
        var h = (condEmbedder(feature1.asType(.float32)) + condEmbedder2(feature2.asType(.float32))).asType(idolStreamType)
        for (repo, block) in zip(contextRepoLayers, contextRefiner) {
            let (cos, sin) = repo(h)
            h = block(h, cos: cos, sin: sin)
        }
        return h
    }

    /// The latent [B, 8192, 16] and camera [B, 1, 5] (float32) at time `t` in [0, 1000].
    public func stream(latent: MLXArray, camera: MLXArray, t: Float) -> Stream {
        let batch = latent.dim(0)
        let timeEmbedding = tEmbedder(MLXArray(Array(repeating: t, count: batch)))
        let modulation = adaLNModulation(silu(timeEmbedding)).asType(idolStreamType)
        var h = inputLayer(latent.asType(.float32)).asType(idolStreamType)
        h = h + positionEmbedding().asType(idolStreamType)
        for (repo, block) in zip(noiseRepoLayers, noiseRefiner) {
            let (cos, sin) = repo(h)
            h = block(h, modulation: modulation, cos: cos, sin: sin)
        }
        let camera = camRefiner(camera.asType(.float32)).asType(idolStreamType)
        return Stream(latent: h, camera: camera, modulation: modulation, timeEmbedding: timeEmbedding)
    }

    /// The joint blocks over [latent, picture, camera] and the heads: the velocity of the latent [B, 8192, 16] and of
    /// the camera [B, 1, 5], float32. `inspect` sees the inner states (`rope_block0`, `block0`, `block_last`).
    public func joint(_ stream: Stream, context: MLXArray, inspect: ((String, MLXArray) -> Void)? = nil) -> (latent: MLXArray, camera: MLXArray) {
        joint(stream, context: context, mask: nil, inspect: inspect)
    }

    /// The joint blocks with the unconditional branch's picture tokens collapsed to two (`collapsed(_:)`).
    public func joint(_ stream: Stream, collapsed: Collapsed) -> (latent: MLXArray, camera: MLXArray) {
        joint(stream, context: collapsed.tokens, mask: collapsed.mask(latentTokens: stream.latent.dim(1)), inspect: nil)
    }

    func joint(_ stream: Stream, context: MLXArray, mask: MLXArray?, inspect: ((String, MLXArray) -> Void)?) -> (latent: MLXArray, camera: MLXArray) {
        let count = stream.latent.dim(1)
        var h = concatenated([stream.latent, context, stream.camera], axis: 1)
        for (index, (repo, block)) in zip(repoLayers, blocks).enumerated() {
            let (cos, sin) = repo(h)
            if index == 0 { inspect?("rope_block0", stacked([cos, sin], axis: -1)) }
            h = block(h, modulation: stream.modulation, cos: cos, sin: sin, mask: mask)
            if index == 0 { inspect?("block0", h) }
        }
        inspect?("block_last", h)
        let length = h.dim(1)
        let shiftScale = shiftTable + stream.timeEmbedding.expandedDimensions(axis: 1)
        let (shift, scale) = (shiftScale[0..., 0 ..< 1], shiftScale[0..., 1 ..< 2])
        func head(_ x: MLXArray, _ layer: Linear) -> MLXArray {
            let normed = MLXFast.layerNorm(x.asType(.float32), weight: nil, bias: nil, eps: 1e-5)
            return layer(normed * (1 + scale) + shift)
        }
        return (head(h[0..., 0 ..< count], outLayer), head(h[0..., (length - 1)...], camOutLayer))
    }

    /// The unconditional branch's picture tokens are all alike (zero features, and every layer treats alike tokens
    /// alike: a token's rotation comes from its own features), so the joint blocks can see them as two tokens whose keys
    /// stand for all of them: additive biases b₁, b₂ with e^b₁ + e^b₂ equal to their count as closely as float16
    /// allows (a single key's ln 4101 is off by up to 0.4% in float16), as mlx-serve's `keyBias`. 8,195 tokens instead
    /// of 12,294, the same velocity to float16's rounding.
    public struct Collapsed {
        public let tokens: MLXArray
        public let biases: (Float16, Float16)

        /// [1, 1, 1, latent + 2 + 1]: zero for every key but the two picture tokens'.
        func mask(latentTokens: Int) -> MLXArray {
            var row = [Float16](repeating: 0, count: latentTokens + 3)
            row[latentTokens] = biases.0
            row[latentTokens + 1] = biases.1
            return MLXArray(row).reshaped([1, 1, 1, row.count])
        }
    }

    /// `context` (the unconditional branch's refined picture tokens, all alike) as two tokens.
    public func collapsed(_ context: MLXArray) -> Collapsed {
        let token = context[0..., 0 ..< 1]
        return Collapsed(tokens: concatenated([token, token], axis: 1), biases: Self.keyBiases(count: context.dim(1)))
    }

    /// Two float16 biases whose exponentials add up to `count` as closely as float16 allows.
    static func keyBiases(count: Int) -> (Float16, Float16) {
        let n = Double(count)
        var best: (Float16, Float16) = (0, 0)
        var bestError = Double.infinity
        var first = Float16(log(n / 2))
        for _ in 0 ..< 64 { first = first.nextDown }
        for _ in 0 ..< 128 {
            first = first.nextUp
            let rest = n - exp(Double(first))
            guard rest > 0 else { break }
            let second = Float16(log(rest))
            let error = abs(exp(Double(first)) + exp(Double(second)) - n)
            if error < bestError { (best, bestError) = ((first, second), error) }
        }
        return best
    }

    /// One whole pass, as idol-sdk's `forward`.
    public func callAsFunction(latent: MLXArray, camera: MLXArray, t: Float, feature1: MLXArray, feature2: MLXArray) -> (latent: MLXArray, camera: MLXArray) {
        joint(stream(latent: latent, camera: camera, t: t), context: context(feature1: feature1, feature2: feature2))
    }
}
