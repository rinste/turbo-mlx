import Foundation
import MLX
import MLXNN

// SenseNova-U1.5 works on pixels, with no VAE: a patch embedding turns the image into tokens
// (`NEOVisionEmbeddings`, one for pictures it reads and one for the image it generates), and a
// convolutional head turns the generation stack's output back into pixels (`ConvDecoder`). Images
// are channels last here, [B, H, W, 3], where the reference is channels first.

/// `NEOVisionEmbeddings`: 16 × 16 patches into features (a convolution and GELU), a 2D rotation of
/// the features by the patch's column (first half) and row (second half), then 2 × 2 patches into
/// one token of the backbone's width.
public final class SenseNovaVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "dense_embedding") var denseEmbedding: Conv2d

    let config: SenseNovaConfig

    public init(config: SenseNovaConfig) {
        self.config = config
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: config.visionHiddenSize,
            kernelSize: .init(config.patchSize), stride: .init(config.patchSize)
        )
        _denseEmbedding.wrappedValue = Conv2d(
            inputChannels: config.visionHiddenSize, outputChannels: config.hiddenSize,
            kernelSize: .init(config.mergeSize), stride: .init(config.mergeSize)
        )
        super.init()
    }

    /// [1, H, W, 3] (normalized as the model expects) → [1, (H/32)·(W/32), hidden], row by row.
    public func callAsFunction(_ image: MLXArray) -> MLXArray {
        let patches = gelu(patchEmbedding(image))
        let rotated = rotate(patches).asType(patchEmbedding.weight.dtype)
        let tokens = denseEmbedding(rotated)
        return tokens.reshaped([tokens.shape[0], tokens.shape[1] * tokens.shape[2], tokens.shape[3]])
    }

    /// `apply_2d_rotary_pos_emb` in float32: the first half of the features rotated by the column,
    /// the second by the row, in interleaved pairs (even, odd).
    func rotate(_ patches: MLXArray) -> MLXArray {
        let (batch, rows, columns, features) = (patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3])
        let part = features / 2
        let exponents = MLXArray(stride(from: 0, to: part, by: 2).map { Float($0) / Float(part) })
        let invFreq = 1 / pow(Float(config.visionRopeTheta), exponents)
        func angles(_ count: Int) -> MLXArray {
            MLXArray((0 ..< count).map { Float($0) }).reshaped([count, 1]) * invFreq.reshaped([1, part / 2])
        }
        let columnAngles = angles(columns).reshaped([1, 1, columns, part / 2])
        let rowAngles = angles(rows).reshaped([1, rows, 1, part / 2])
        let x = patches.asType(.float32)

        func rotatePart(_ x: MLXArray, _ angles: MLXArray) -> MLXArray {
            let pairs = x.reshaped([batch, rows, columns, part / 2, 2])
            let x1 = pairs[.ellipsis, 0]
            let x2 = pairs[.ellipsis, 1]
            let c = cos(angles)
            let s = sin(angles)
            return stacked([x1 * c - x2 * s, x1 * s + x2 * c], axis: -1).reshaped([batch, rows, columns, part])
        }
        return concatenated([rotatePart(x[.ellipsis, 0 ..< part], columnAngles), rotatePart(x[.ellipsis, part...], rowAngles)], axis: -1)
    }
}

/// `TimestepEmbedder`: a sinusoidal embedding of a scalar in [0, 1] (256 frequencies, cosines
/// first) through two linears with SiLU between them (`mlp.0`, `mlp.1` in the pack).
public final class SenseNovaTimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: [Linear]

    static let frequencyEmbeddingSize = 256

    public init(hiddenSize: Int) {
        _mlp.wrappedValue = [
            Linear(Self.frequencyEmbeddingSize, hiddenSize, bias: true),
            Linear(hiddenSize, hiddenSize, bias: true),
        ]
        super.init()
    }

    /// [1, hidden] for the value `t`, in the linears' dtype.
    public func callAsFunction(_ t: Float) -> MLXArray {
        let half = Self.frequencyEmbeddingSize / 2
        let freqs = exp(-Float(log(10000.0)) * MLXArray(0 ..< Int32(half)).asType(.float32) / Float(half))
        let args = MLXArray([t]).reshaped([1, 1]) * freqs.reshaped([1, half])
        // `.to(self.mlp[0].weight.dtype)`: the bias has it too, and keeps it if the weight is quantized.
        let embedding = concatenated([cos(args), sin(args)], axis: -1).asType(mlp[0].bias?.dtype ?? .float32)
        return mlp[1](silu(mlp[0](embedding)))
    }
}

/// `ConvDecoder`: the generation stack's tokens [B, h, w, hidden] into pixels [B, 32h, 32w, 3]:
/// pixel shuffle ×2, a 3 × 3 convolution and GELU, pixel shuffle ×2, a 3 × 3 convolution to
/// 3 · 8 · 8 channels, pixel shuffle ×8.
public final class SenseNovaPixelHead: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    public init(hiddenSize: Int, width: Int = 1024) {
        _conv1.wrappedValue = Conv2d(inputChannels: hiddenSize / 4, outputChannels: width, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv2d(inputChannels: width / 4, outputChannels: 192, kernelSize: 3, padding: 1)
        super.init()
    }

    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let x = gelu(conv1(Self.pixelShuffle(tokens, 2)))
        return Self.pixelShuffle(conv2(Self.pixelShuffle(x, 2)), 8)
    }

    /// `nn.PixelShuffle(r)` channels last: channel c·r² + i·r + j of pixel (y, x) goes to channel c
    /// of pixel (y·r + i, x·r + j).
    static func pixelShuffle(_ x: MLXArray, _ r: Int) -> MLXArray {
        let (batch, height, width, channels) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let out = channels / (r * r)
        return x.reshaped([batch, height, width, out, r, r])
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped([batch, height * r, width * r, out])
    }
}

/// The modules outside the backbone, keyed as the checkpoint's `fm_modules.*`.
public final class SenseNovaFlowModules: Module {
    @ModuleInfo(key: "vision_model_mot_gen") var visionModel: SenseNovaVisionModel
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: SenseNovaTimestepEmbedder
    @ModuleInfo(key: "noise_scale_embedder") var noiseScaleEmbedder: SenseNovaTimestepEmbedder?
    @ModuleInfo(key: "fm_head") var head: SenseNovaPixelHead

    public init(config: SenseNovaConfig) {
        _visionModel.wrappedValue = SenseNovaVisionModel(config: config)
        _timestepEmbedder.wrappedValue = SenseNovaTimestepEmbedder(hiddenSize: config.hiddenSize)
        _noiseScaleEmbedder.wrappedValue = config.addNoiseScaleEmbedding ? SenseNovaTimestepEmbedder(hiddenSize: config.hiddenSize) : nil
        _head.wrappedValue = SenseNovaPixelHead(hiddenSize: config.hiddenSize)
        super.init()
    }
}

/// `NEOVisionModel`: its embeddings are all there is.
public final class SenseNovaVisionModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SenseNovaVisionEmbeddings

    public init(config: SenseNovaConfig) {
        _embeddings.wrappedValue = SenseNovaVisionEmbeddings(config: config)
        super.init()
    }

    public func callAsFunction(_ image: MLXArray) -> MLXArray { embeddings(image) }
}
