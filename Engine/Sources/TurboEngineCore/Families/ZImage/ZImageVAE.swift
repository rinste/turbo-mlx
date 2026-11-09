import Foundation
import MLX
import MLXNN

// Z-Image's autoencoder (FLUX.1's 16-channel layout), after mflux's `z_image_vae/*`.
// Activations are channels-last throughout (the reference transposes around every
// convolution, which changes nothing numerically). Unlike FLUX.2's, its residual blocks normalize
// in the activations' dtype; only the attention's norm and the decoder's output norm run in
// float32. The encoder only serves image to image: `ZImageVAE` skips its weights, and
// `ZImageEncoder` is loaded on its own the first time a picture is encoded.

final class ZImageResnetBlock2D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int) {
        _norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: inChannels, eps: 1e-6, pytorchCompatible: true)
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        _norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels, eps: 1e-6, pytorchCompatible: true)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        _convShortcut.wrappedValue = inChannels != outChannels
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, stride: 1, padding: 0)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        let residual = convShortcut.map { $0(x) } ?? x
        return residual + h
    }
}

final class ZImageVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    init(channels: Int) {
        _groupNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
        _toQ.wrappedValue = Linear(channels, channels)
        _toK.wrappedValue = Linear(channels, channels)
        _toV.wrappedValue = Linear(channels, channels)
        _toOut.wrappedValue = [Linear(channels, channels)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let normed = groupNorm(x.asType(.float32)).asType(modelPrecision)
        let q = toQ(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        let k = toK(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        let v = toV(normed).reshaped([batch, height * width, 1, channels]).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(channels).squareRoot(), mask: nil
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, height, width, channels])
        return x + toOut[0](merged)
    }
}

final class ZImageUNetMidBlock: Module {
    @ModuleInfo(key: "attentions") var attentions: [ZImageVAEAttention]
    @ModuleInfo(key: "resnets") var resnets: [ZImageResnetBlock2D]

    init(channels: Int) {
        _attentions.wrappedValue = [ZImageVAEAttention(channels: channels)]
        _resnets.wrappedValue = [
            ZImageResnetBlock2D(inChannels: channels, outChannels: channels),
            ZImageResnetBlock2D(inChannels: channels, outChannels: channels),
        ]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

final class ZImageUpSampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Nearest-neighbour doubling of height and width.
        conv(repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2))
    }
}

final class ZImageUpDecoderBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [ZImageResnetBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [ZImageUpSampler]

    init(inChannels: Int, outChannels: Int, numLayers: Int = 3, addUpsample: Bool) {
        _resnets.wrappedValue = (0 ..< numLayers).map { index in
            ZImageResnetBlock2D(inChannels: index == 0 ? inChannels : outChannels, outChannels: outChannels)
        }
        _upsamplers.wrappedValue = addUpsample ? [ZImageUpSampler(channels: outChannels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        for upsampler in upsamplers { h = upsampler(h) }
        return h
    }
}

/// mflux wraps the first and last convolutions and the output norm in modules of their own.
final class ZImageConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class ZImageConvNormOut: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    init(channels: Int) {
        _norm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(x.asType(.float32)).asType(modelPrecision)
    }
}

final class ZImageDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: ZImageConvLayer
    @ModuleInfo(key: "mid_block") var midBlock: ZImageUNetMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [ZImageUpDecoderBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: ZImageConvNormOut
    @ModuleInfo(key: "conv_out") var convOut: ZImageConvLayer

    /// `blockOutChannels` first stage first ([128, 256, 512, 512] for the real model): the decoder
    /// walks them from the last, doubling the size after every stage but the first.
    init(latentChannels: Int = 16, outChannels: Int = 3, blockOutChannels: [Int]) {
        let reversedChannels = Array(blockOutChannels.reversed())
        _convIn.wrappedValue = ZImageConvLayer(inChannels: latentChannels, outChannels: reversedChannels[0])
        _midBlock.wrappedValue = ZImageUNetMidBlock(channels: reversedChannels[0])
        _upBlocks.wrappedValue = reversedChannels.enumerated().map { index, outChannels in
            ZImageUpDecoderBlock(
                inChannels: index == 0 ? outChannels : reversedChannels[index - 1],
                outChannels: outChannels,
                addUpsample: index != reversedChannels.count - 1
            )
        }
        _convNormOut.wrappedValue = ZImageConvNormOut(channels: blockOutChannels[0])
        _convOut.wrappedValue = ZImageConvLayer(inChannels: blockOutChannels[0], outChannels: outChannels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = midBlock(convIn(x))
        for block in upBlocks { h = block(h) }
        return convOut(silu(convNormOut(h)))
    }
}

// MARK: Encoder

/// mflux's encoder `DownSampler`: the bottom and right edges padded by one, then a 3 × 3
/// convolution of stride 2 without padding of its own.
final class ZImageDownSampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [0, [0, 1], [0, 1], 0]))
    }
}

final class ZImageDownEncoderBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [ZImageResnetBlock2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [ZImageDownSampler]

    init(inChannels: Int, outChannels: Int, numLayers: Int = 2, addDownsample: Bool) {
        _resnets.wrappedValue = (0 ..< numLayers).map { index in
            ZImageResnetBlock2D(inChannels: index == 0 ? inChannels : outChannels, outChannels: outChannels)
        }
        _downsamplers.wrappedValue = addDownsample ? [ZImageDownSampler(channels: outChannels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        for downsampler in downsamplers { h = downsampler(h) }
        return h
    }
}

/// The encoder's first and last convolutions: mflux names them `conv2d` where the decoder's are `conv`.
final class ZImageEncoderConv: Module {
    @ModuleInfo(key: "conv2d") var conv: Conv2d
    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// The encoder's output norm runs in the activations' dtype, unlike the decoder's.
final class ZImageEncoderNormOut: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    init(channels: Int) {
        _norm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(x) }
}

/// The encoder half of the VAE, for image to image: a picture in [-1, 1] to the mean and
/// log-variance of its 16 latent channels, halving the size after every stage but the last. The
/// picture arrives in float32, as mflux hands it over, and MLX keeps the stream there against the
/// bf16 weights; only the attention works in the weights' precision, as in the decoder.
public final class ZImageEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: ZImageEncoderConv
    @ModuleInfo(key: "down_blocks") var downBlocks: [ZImageDownEncoderBlock]
    @ModuleInfo(key: "mid_block") var midBlock: ZImageUNetMidBlock
    @ModuleInfo(key: "conv_norm_out") var convNormOut: ZImageEncoderNormOut
    @ModuleInfo(key: "conv_out") var convOut: ZImageEncoderConv

    /// `blockOutChannels` first stage first, as the decoder takes them.
    public init(inChannels: Int = 3, latentChannels: Int = 16, blockOutChannels: [Int] = [128, 256, 512, 512]) {
        _convIn.wrappedValue = ZImageEncoderConv(inChannels: inChannels, outChannels: blockOutChannels[0])
        _downBlocks.wrappedValue = blockOutChannels.enumerated().map { index, outChannels in
            ZImageDownEncoderBlock(
                inChannels: index == 0 ? blockOutChannels[0] : blockOutChannels[index - 1],
                outChannels: outChannels,
                addDownsample: index != blockOutChannels.count - 1
            )
        }
        let last = blockOutChannels[blockOutChannels.count - 1]
        _midBlock.wrappedValue = ZImageUNetMidBlock(channels: last)
        _convNormOut.wrappedValue = ZImageEncoderNormOut(channels: last)
        _convOut.wrappedValue = ZImageEncoderConv(inChannels: last, outChannels: 2 * latentChannels)
        super.init()
    }

    /// The VAE checkpoint's encoder tensors, without their `encoder.` prefix.
    public static func weights(_ vae: [String: MLXArray]) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for (key, value) in vae where key.hasPrefix("encoder.") { weights[String(key.dropFirst("encoder.".count))] = value }
        return weights
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for block in downBlocks { h = block(h) }
        h = midBlock(h)
        return convOut(silu(convNormOut(h)))
    }

    /// [B, H, W, 3] in [-1, 1] → [B, H/8, W/8, 16] latents as the transformer reads them
    /// (`VAE.encode`): the mean, shifted and scaled by the checkpoint's factors.
    public func encode(_ image: MLXArray) -> MLXArray {
        let moments = self(image)
        let mean = moments[.ellipsis, 0 ..< ZImageVAE.latentChannels]
        return (mean - ZImageVAE.shiftFactor) * ZImageVAE.scalingFactor
    }
}

// MARK: VAE

public final class ZImageVAE: Module {
    @ModuleInfo(key: "decoder") var decoder: ZImageDecoder

    static let scalingFactor: Float = 0.3611
    static let shiftFactor: Float = 0.1159
    public static let latentChannels = 16
    public static let spatialScale = 8

    public init(blockOutChannels: [Int] = [128, 256, 512, 512]) {
        _decoder.wrappedValue = ZImageDecoder(blockOutChannels: blockOutChannels)
        super.init()
    }

    /// The encoder is loaded apart (`ZImageEncoder`), when a picture is first encoded.
    public static func ignoresKey(_ key: String) -> Bool {
        key.hasPrefix("encoder.")
    }

    /// [B, h, w, 16] channels-last latents → [B, H, W, 3] in [-1, 1]: unscaled with the
    /// checkpoint's factors, then decoded.
    public func decode(_ latents: MLXArray) -> MLXArray {
        decoder(latents / Self.scalingFactor + Self.shiftFactor)
    }
}
