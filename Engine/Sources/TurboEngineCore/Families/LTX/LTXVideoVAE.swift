import Foundation
import MLX
import MLXNN

// LTX-2's video autoencoder, after dgrauet's `model/video_vae/`: a 3D convolutional decoder
// (latent frames × 8, pixels × 32, through pixel shuffles) and the causal encoder used to turn a
// reference image into the latent of the first frame. Activations are channels-last,
// [B, frames, height, width, channels], as MLX's convolutions want them. The pack stores both
// halves in bfloat16 (`vae_decoder.safetensors`, `vae_encoder.safetensors`); the lists mix two
// kinds of blocks, so each list element holds whichever of the two its keys name.

/// `pixel_norm`: RMS over channels without a weight.
private func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
    unweightedRMSNorm(x, eps: eps)
}

/// `Conv3dBlock`: a 3D convolution padded by hand — the first frame repeated in front (causal),
/// or the first and last frames repeated on both sides; zeros around each frame.
final class LTXConv3d: Module {
    @ModuleInfo(key: "conv") var conv: Conv3d

    let temporalKernel: Int
    let causal: Bool
    let spatialPadding: Int

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int = 3, padding: Int = 1, causal: Bool) {
        temporalKernel = kernel
        self.causal = causal
        spatialPadding = padding
        _conv.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: .init(kernel), padding: .init(0))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if causal {
            if temporalKernel > 1 {
                x = concatenated([repeated(x[0..., 0 ..< 1], count: temporalKernel - 1, axis: 1), x], axis: 1)
            }
        } else {
            let pad = (temporalKernel - 1) / 2
            if pad > 0 {
                let frames = x.shape[1]
                x = concatenated([
                    repeated(x[0..., 0 ..< 1], count: pad, axis: 1), x, repeated(x[0..., (frames - 1) ..< frames], count: pad, axis: 1),
                ], axis: 1)
            }
        }
        if spatialPadding > 0 {
            let p = spatialPadding
            x = padded(x, widths: [.init(0), .init(0), .init((p, p)), .init((p, p)), .init(0)])
        }
        return conv(x)
    }
}

final class LTXResBlock3d: Module {
    @ModuleInfo(key: "conv1") var conv1: LTXConv3d
    @ModuleInfo(key: "conv2") var conv2: LTXConv3d

    init(channels: Int, causal: Bool) {
        _conv1.wrappedValue = LTXConv3d(channels, channels, causal: causal)
        _conv2.wrappedValue = LTXConv3d(channels, channels, causal: causal)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = conv1(silu(pixelNorm(x)))
        return conv2(silu(pixelNorm(h))) + x
    }
}

/// One element of `up_blocks` / `down_blocks`: a stage of residual blocks, or a resampling
/// convolution (`conv`), whose rearrangement the owner applies.
final class LTXVAEBlock: Module {
    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXResBlock3d]?
    @ModuleInfo(key: "conv") var conv: LTXConv3d?

    /// A residual stage.
    init(channels: Int, blocks: Int, causal: Bool) {
        _resBlocks.wrappedValue = (0 ..< blocks).map { _ in LTXResBlock3d(channels: channels, causal: causal) }
        _conv.wrappedValue = nil
        super.init()
    }

    /// A resampling convolution.
    init(convolution inChannels: Int, _ outChannels: Int, causal: Bool) {
        _resBlocks.wrappedValue = nil
        _conv.wrappedValue = LTXConv3d(inChannels, outChannels, causal: causal)
        super.init()
    }

    func residual(_ x: MLXArray) -> MLXArray {
        var x = x
        for block in resBlocks ?? [] { x = block(x) }
        return x
    }
}

final class LTXChannelStatistics: Module {
    @ParameterInfo(key: "mean") var mean: MLXArray
    @ParameterInfo(key: "std") var std: MLXArray

    init(channels: Int) {
        _mean.wrappedValue = MLXArray.zeros([channels])
        _std.wrappedValue = MLXArray.ones([channels])
        super.init()
    }
}

final class LTXEncoderStatistics: Module {
    @ParameterInfo(key: "mean_of_means") var meanOfMeans: MLXArray
    @ParameterInfo(key: "std_of_means") var stdOfMeans: MLXArray

    init(channels: Int) {
        _meanOfMeans.wrappedValue = MLXArray.zeros([channels])
        _stdOfMeans.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    /// `normalize_latent` / `denormalize_latent` on channels-last latents.
    func normalize(_ latent: MLXArray) -> MLXArray { (latent - meanOfMeans) / stdOfMeans }
    func denormalize(_ latent: MLXArray) -> MLXArray { latent * stdOfMeans + meanOfMeans }
}

// MARK: - Rearrangements

/// `pixel_shuffle_3d`: [B, D, H, W, C·t·s·s] → [B, D·t, H·s, W·s, C], channels outermost.
func ltxPixelShuffle3d(_ x: MLXArray, spatial s: Int, temporal t: Int) -> MLXArray {
    let (b, d, h, w, total) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    let c = total / (s * s * t)
    return x.reshaped([b, d, h, w, c, t, s, s]).transposed(0, 1, 5, 2, 6, 3, 7, 4).reshaped([b, d * t, h * s, w * s, c])
}

/// `unpatchify_spatial`: [B, F, H, W, C·p·p] → [B, F, H·p, W·p, C], the width factor first.
private func unpatchifySpatial(_ x: MLXArray, patch p: Int) -> MLXArray {
    let (b, f, h, w, total) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    let c = total / (p * p)
    return x.reshaped([b, f, h, w, c, p, p]).transposed(0, 1, 2, 6, 3, 5, 4).reshaped([b, f, h * p, w * p, c])
}

/// `patchify_spatial`: [B, F, H, W, C] → [B, F, H/p, W/p, C·p·p].
private func patchifySpatial(_ x: MLXArray, patch p: Int) -> MLXArray {
    let (b, f, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    return x.reshaped([b, f, h / p, p, w / p, p, c]).transposed(0, 1, 2, 4, 6, 5, 3).reshaped([b, f, h / p, w / p, c * p * p])
}

/// `space_to_depth` with strides (t, h, w).
private func spaceToDepth(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
    let (b, d, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    let (st, sh, sw) = stride
    return x.reshaped([b, d / st, st, h / sh, sh, w / sw, sw, c]).transposed(0, 1, 3, 5, 7, 2, 4, 6)
        .reshaped([b, d / st, h / sh, w / sw, c * st * sh * sw])
}

// MARK: - Decoder

final class LTXVideoDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: LTXConv3d
    @ModuleInfo(key: "up_blocks") var upBlocks: [LTXVAEBlock]
    @ModuleInfo(key: "conv_out") var convOut: LTXConv3d
    @ModuleInfo(key: "per_channel_statistics") var statistics: LTXChannelStatistics

    /// (spatial, temporal) factors of the four resampling blocks, at indices 1, 3, 5, 7.
    private static let upsampling = [(2, 2), (2, 2), (1, 2), (2, 1)]

    /// LTX-2.3's decoder is not causal: its convolutions repeat the first and last frames.
    init(causal: Bool = false) {
        _convIn.wrappedValue = LTXConv3d(128, 1024, causal: causal)
        _upBlocks.wrappedValue = [
            LTXVAEBlock(channels: 1024, blocks: 2, causal: causal),
            LTXVAEBlock(convolution: 1024, 4096, causal: causal),
            LTXVAEBlock(channels: 512, blocks: 2, causal: causal),
            LTXVAEBlock(convolution: 512, 4096, causal: causal),
            LTXVAEBlock(channels: 512, blocks: 4, causal: causal),
            LTXVAEBlock(convolution: 512, 512, causal: causal),
            LTXVAEBlock(channels: 256, blocks: 6, causal: causal),
            LTXVAEBlock(convolution: 256, 512, causal: causal),
            LTXVAEBlock(channels: 128, blocks: 4, causal: causal),
        ]
        _convOut.wrappedValue = LTXConv3d(128, 48, causal: causal)
        _statistics.wrappedValue = LTXChannelStatistics(channels: 128)
        super.init()
    }

    /// Latent [B, 128, F, H, W] → pixels [B, 3, 8(F−1)+1, 32H, 32W] in [-1, 1], in the latent's
    /// type. `evaluateStages` evaluates after each resampling (the tiled decode does, to free the
    /// previous stage before the next, larger one).
    func decode(_ latent: MLXArray, evaluateStages: Bool = false) -> MLXArray {
        let outputType = latent.dtype
        let weightType = convIn.conv.weight.dtype
        var x = latent.asType(weightType).transposed(0, 2, 3, 4, 1)
        x = x * statistics.std + statistics.mean
        x = convIn(x)
        var upsample = 0
        for block in upBlocks {
            if let conv = block.conv {
                let (s, t) = Self.upsampling[upsample]
                x = ltxPixelShuffle3d(conv(x), spatial: s, temporal: t)
                // The first frame after a temporal upsampling is the causal padding's echo.
                if t > 1 { x = x[0..., 1...] }
                upsample += 1
                if evaluateStages { eval(x) }
            } else {
                x = block.residual(x)
            }
        }
        x = convOut(silu(pixelNorm(x)))
        x = unpatchifySpatial(x, patch: 4)
        return x.transposed(0, 4, 1, 2, 3).asType(outputType)
    }

    /// `decode`, keeping every stage (for verify).
    func stages(_ latent: MLXArray) -> [(String, MLXArray)] {
        var result: [(String, MLXArray)] = []
        var x = latent.asType(convIn.conv.weight.dtype).transposed(0, 2, 3, 4, 1)
        x = x * statistics.std + statistics.mean
        result.append(("denorm", x))
        x = convIn(x)
        result.append(("conv_in", x))
        var upsample = 0
        for (index, block) in upBlocks.enumerated() {
            if let conv = block.conv {
                let (s, t) = Self.upsampling[upsample]
                x = ltxPixelShuffle3d(conv(x), spatial: s, temporal: t)
                if t > 1 { x = x[0..., 1...] }
                upsample += 1
            } else {
                x = block.residual(x)
            }
            result.append(("up_\(index)", x))
        }
        x = convOut(silu(pixelNorm(x)))
        result.append(("conv_out", x))
        return result
    }

    static func weights(_ tensors: [String: MLXArray], prefix: String = "vae_decoder.") -> [String: MLXArray] {
        stripping(prefix, from: tensors)
    }
}

// MARK: - Encoder

/// `SpaceToDepthDownsample`: a convolution rearranged into depth, plus the input rearranged and
/// averaged over channel groups as the skip.
private func spaceToDepthDownsample(_ x: MLXArray, conv: LTXConv3d, stride: (Int, Int, Int), inChannels: Int, outChannels: Int) -> MLXArray {
    var x = x
    if stride.0 == 2 { x = concatenated([x[0..., 0 ..< 1], x], axis: 1) }
    let groupSize = inChannels * stride.1 * stride.2 * stride.0 / outChannels
    var skip = spaceToDepth(x, stride: stride)
    if groupSize > 1 {
        let (b, d, h, w, total) = (skip.shape[0], skip.shape[1], skip.shape[2], skip.shape[3], skip.shape[4])
        skip = skip.reshaped([b, d, h, w, total / groupSize, groupSize]).mean(axis: -1)
    }
    return spaceToDepth(conv(x), stride: stride) + skip
}

final class LTXVideoEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: LTXConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [LTXVAEBlock]
    @ModuleInfo(key: "conv_out") var convOut: LTXConv3d
    @ModuleInfo(key: "per_channel_statistics") var statistics: LTXEncoderStatistics

    /// (stride, in, out) of the four downsamplings, at indices 1, 3, 5, 7.
    private static let downsampling: [((Int, Int, Int), Int, Int)] = [
        ((1, 2, 2), 128, 256), ((2, 1, 1), 256, 512), ((2, 2, 2), 512, 1024), ((2, 2, 2), 1024, 1024),
    ]

    override init() {
        _convIn.wrappedValue = LTXConv3d(48, 128, causal: true)
        var blocks: [LTXVAEBlock] = []
        let stages = [(128, 4), (256, 6), (512, 4), (1024, 2), (1024, 2)]
        for (index, stage) in stages.enumerated() {
            blocks.append(LTXVAEBlock(channels: stage.0, blocks: stage.1, causal: true))
            if index < Self.downsampling.count {
                let (stride, inChannels, outChannels) = Self.downsampling[index]
                blocks.append(LTXVAEBlock(convolution: inChannels, outChannels / (stride.0 * stride.1 * stride.2), causal: true))
            }
        }
        _downBlocks.wrappedValue = blocks
        _convOut.wrappedValue = LTXConv3d(1024, 129, causal: true)
        _statistics.wrappedValue = LTXEncoderStatistics(channels: 128)
        super.init()
    }

    /// Pixels [B, 3, F, H, W] in [-1, 1] → normalized latent [B, 128, F', H/32, W/32].
    func encode(_ pixels: MLXArray) -> MLXArray {
        var x = patchifySpatial(pixels.transposed(0, 2, 3, 4, 1), patch: 4)
        x = convIn(x)
        var downsample = 0
        for block in downBlocks {
            if let conv = block.conv {
                let (stride, inChannels, outChannels) = Self.downsampling[downsample]
                x = spaceToDepthDownsample(x, conv: conv, stride: stride, inChannels: inChannels, outChannels: outChannels)
                downsample += 1
            } else {
                x = block.residual(x)
            }
        }
        x = convOut(silu(pixelNorm(x)))
        x = statistics.normalize(x[.ellipsis, 0 ..< 128])
        return x.transposed(0, 4, 1, 2, 3)
    }

    static func weights(_ tensors: [String: MLXArray], prefix: String = "vae_encoder.") -> [String: MLXArray] {
        remappingStatistics(stripping(prefix, from: tensors))
    }
}

/// Keys under `prefix`, with the prefix removed.
func stripping(_ prefix: String, from tensors: [String: MLXArray]) -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    for (key, value) in tensors where key.hasPrefix(prefix) {
        weights[String(key.dropFirst(prefix.count))] = value
    }
    return weights
}

/// The encoders store their statistics as `_mean_of_means` / `_std_of_means`.
func remappingStatistics(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    for (key, value) in tensors {
        weights[key.replacingOccurrences(of: "._mean_of_means", with: ".mean_of_means")
            .replacingOccurrences(of: "._std_of_means", with: ".std_of_means")] = value
    }
    return weights
}
