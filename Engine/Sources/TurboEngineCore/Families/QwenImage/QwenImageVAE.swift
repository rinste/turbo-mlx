import Foundation
import MLX
import MLXNN

// Qwen-Image's autoencoder (Wan 2.1's causal 3D VAE), decoder only, after mflux's `qwen_vae/*`;
// Ming-Image's is the same network retrained for RGBA with one scaling factor. Activations are
// channels-last [B, T, H, W, C] throughout (the reference transposes around every convolution).
// A still image is one frame, so the temporal up-sampling never runs and its `time_conv`
// weights are skipped when loading.

/// A 3D convolution padded causally in time (twice the padding before, none after).
final class QwenCausalConv3D: Module {
    @ModuleInfo(key: "conv3d") var conv3d: Conv3d
    let padding: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, padding: Int) {
        self.padding = padding
        _conv3d.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: IntOrTriple(kernelSize), stride: 1, padding: 0)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if padding > 0 {
            let p = padding
            x = padded(x, widths: [IntOrPair(0), IntOrPair((2 * p, 0)), IntOrPair((p, p)), IntOrPair((p, p)), IntOrPair(0)])
        }
        return conv3d(x)
    }
}

/// `QwenImageRMSNorm`: the L2 norm over the channels (floored at eps), scaled by √C and a weight.
/// The weight is flat ([C]), as mflux's checkpoints store it (see `QwenImageVAE.weights(_:)`).
final class QwenImageRMSNorm: Module {
    @ParameterInfo var weight: MLXArray
    let eps: Float = 1e-12
    let scale: Float

    init(channels: Int) {
        scale = Float(channels).squareRoot()
        _weight.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let norm = sqrt((x * x).sum(axis: -1, keepDims: true))
        let denominator = maximum(norm, MLXArray(eps).asType(norm.dtype))
        return x / denominator * scale * weight
    }
}

final class QwenResBlock3D: Module {
    @ModuleInfo(key: "norm1") var norm1: QwenImageRMSNorm
    @ModuleInfo(key: "conv1") var conv1: QwenCausalConv3D
    @ModuleInfo(key: "norm2") var norm2: QwenImageRMSNorm
    @ModuleInfo(key: "conv2") var conv2: QwenCausalConv3D
    @ModuleInfo(key: "skip_conv") var skipConv: QwenCausalConv3D?

    init(inChannels: Int, outChannels: Int) {
        _norm1.wrappedValue = QwenImageRMSNorm(channels: inChannels)
        _conv1.wrappedValue = QwenCausalConv3D(inChannels: inChannels, outChannels: outChannels, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = QwenImageRMSNorm(channels: outChannels)
        _conv2.wrappedValue = QwenCausalConv3D(inChannels: outChannels, outChannels: outChannels, kernelSize: 3, padding: 1)
        _skipConv.wrappedValue = inChannels != outChannels
            ? QwenCausalConv3D(inChannels: inChannels, outChannels: outChannels, kernelSize: 1, padding: 0)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        let residual = skipConv.map { $0(x) } ?? x
        return h + residual
    }
}

/// Self-attention over each frame's pixels, one head.
final class QwenAttentionBlock3D: Module {
    @ModuleInfo(key: "norm") var norm: QwenImageRMSNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
    @ModuleInfo(key: "proj") var proj: Conv2d
    let dim: Int

    init(dim: Int) {
        self.dim = dim
        _norm.wrappedValue = QwenImageRMSNorm(channels: dim)
        _toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1, stride: 1, padding: 0)
        _proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1, stride: 1, padding: 0)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, time, height, width, channels) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        let frames = x.reshaped([batch * time, height, width, channels])
        let qkv = toQKV(norm(frames)).reshaped([batch * time, height * width, 3 * channels])
        let parts = qkv.split(parts: 3, axis: -1)
        // Computed as mflux does: the scores in the activations' dtype, then a float32 scale. A
        // decode that starts in bf16 (Ming-Image's) goes on in float32 from here, as it does there.
        let scale = 1 / sqrt(MLXArray(Float(channels)))
        let scores = matmul(parts[0], parts[1].transposed(0, 2, 1)) * scale
        let attended = matmul(softmax(scores, axis: -1), parts[2])
        let merged = proj(attended.reshaped([batch * time, height, width, channels]))
        return merged.reshaped([batch, time, height, width, channels]) + x
    }
}

/// Nearest-neighbour doubling of each frame, then a convolution halving the channels.
final class QwenResample3D: Module {
    @ModuleInfo(key: "resample_conv") var resampleConv: Conv2d

    init(dim: Int) {
        _resampleConv.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, time, height, width, channels) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        var frames = x.reshaped([batch * time, height, width, channels])
        frames = repeated(repeated(frames, count: 2, axis: 1), count: 2, axis: 2)
        let out = resampleConv(frames)
        return out.reshaped([batch, time, out.shape[1], out.shape[2], out.shape[3]])
    }
}

final class QwenUpBlock3D: Module {
    @ModuleInfo(key: "resnets") var resnets: [QwenResBlock3D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [QwenResample3D]

    init(inChannels: Int, outChannels: Int, numResBlocks: Int = 2, upsample: Bool) {
        var resnets: [QwenResBlock3D] = []
        var current = inChannels
        for _ in 0 ..< (numResBlocks + 1) {
            resnets.append(QwenResBlock3D(inChannels: current, outChannels: outChannels))
            current = outChannels
        }
        _resnets.wrappedValue = resnets
        _upsamplers.wrappedValue = upsample ? [QwenResample3D(dim: outChannels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        for upsampler in upsamplers { h = upsampler(h) }
        return h
    }
}

final class QwenMidBlock3D: Module {
    @ModuleInfo(key: "resnets") var resnets: [QwenResBlock3D]
    @ModuleInfo(key: "attentions") var attentions: [QwenAttentionBlock3D]

    init(dim: Int) {
        _resnets.wrappedValue = [QwenResBlock3D(inChannels: dim, outChannels: dim), QwenResBlock3D(inChannels: dim, outChannels: dim)]
        _attentions.wrappedValue = [QwenAttentionBlock3D(dim: dim)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

final class QwenDecoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: QwenCausalConv3D
    @ModuleInfo(key: "mid_block") var midBlock: QwenMidBlock3D
    @ModuleInfo(key: "up_block0") var upBlock0: QwenUpBlock3D
    @ModuleInfo(key: "up_block1") var upBlock1: QwenUpBlock3D
    @ModuleInfo(key: "up_block2") var upBlock2: QwenUpBlock3D
    @ModuleInfo(key: "up_block3") var upBlock3: QwenUpBlock3D
    @ModuleInfo(key: "norm_out") var normOut: QwenImageRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: QwenCausalConv3D

    /// Stages of `baseDim`, 2× and 4× channels (96, 192, 384 in the model).
    init(latentChannels: Int = 16, outChannels: Int, baseDim: Int) {
        let (d1, d2, d4) = (baseDim, 2 * baseDim, 4 * baseDim)
        _convIn.wrappedValue = QwenCausalConv3D(inChannels: latentChannels, outChannels: d4, kernelSize: 3, padding: 1)
        _midBlock.wrappedValue = QwenMidBlock3D(dim: d4)
        _upBlock0.wrappedValue = QwenUpBlock3D(inChannels: d4, outChannels: d4, upsample: true)
        _upBlock1.wrappedValue = QwenUpBlock3D(inChannels: d2, outChannels: d4, upsample: true)
        _upBlock2.wrappedValue = QwenUpBlock3D(inChannels: d2, outChannels: d2, upsample: true)
        _upBlock3.wrappedValue = QwenUpBlock3D(inChannels: d1, outChannels: d1, upsample: false)
        _normOut.wrappedValue = QwenImageRMSNorm(channels: d1)
        _convOut.wrappedValue = QwenCausalConv3D(inChannels: d1, outChannels: outChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = midBlock(convIn(x))
        h = upBlock3(upBlock2(upBlock1(upBlock0(h))))
        return convOut(silu(normOut(h)))
    }
}

public final class QwenImageVAE: Module {
    /// How the transformer's latents relate to the decoder's: Qwen-Image's per-channel
    /// statistics, or Ming-Image's single factor.
    public enum Normalization: Equatable, Sendable {
        case meanStd
        case scale(Float)
    }

    @ModuleInfo(key: "decoder") var decoder: QwenDecoder3D
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: QwenCausalConv3D

    public static let latentChannels = 16
    public static let spatialScale = 8
    static let latentsMean: [Float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    static let latentsStd: [Float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]

    public let normalization: Normalization

    public init(outChannels: Int = 3, baseDim: Int = 96, normalization: Normalization = .meanStd) {
        self.normalization = normalization
        _decoder.wrappedValue = QwenDecoder3D(latentChannels: Self.latentChannels, outChannels: outChannels, baseDim: baseDim)
        _postQuantConv.wrappedValue = QwenCausalConv3D(inChannels: Self.latentChannels, outChannels: Self.latentChannels, kernelSize: 1, padding: 0)
        super.init()
    }

    /// The encoder, its quantization convolution and the temporal up-sampling convolutions are
    /// not part of this still-image, decoder-only port.
    public static func ignoresKey(_ key: String) -> Bool {
        key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") || key.contains(".time_conv.")
    }

    /// The checkpoint's tensors with the norms' weights flat. mflux's checkpoints store them so
    /// ([C], `reshape_gamma_to_1d`); one saved straight from mflux's modules, as the fixtures
    /// are, keeps [C, 1, 1(, 1)]. mflux takes either and reshapes the weight where it uses it.
    public static func weights(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        var weights = tensors
        for (key, value) in tensors where value.ndim > 1 {
            let path = key.split(separator: ".")
            if path.count >= 2, path[path.count - 1] == "weight", path[path.count - 2].hasPrefix("norm") {
                weights[key] = value.reshaped([-1])
            }
        }
        return weights
    }

    /// [B, h, w, 16] channels-last latents (as the transformer left them) → [B, H, W, C] in [-1, 1].
    public func decode(_ latents: MLXArray) -> MLXArray {
        var x = latents.expandedDimensions(axis: 1)
        switch normalization {
        case .meanStd:
            let mean = MLXArray(Self.latentsMean).reshaped([1, 1, 1, 1, Self.latentChannels])
            let std = MLXArray(Self.latentsStd).reshaped([1, 1, 1, 1, Self.latentChannels])
            x = x * std + mean
        case .scale(let factor):
            x = x / factor
        }
        return decoder(postQuantConv(x)).squeezed(axis: 1)
    }
}
