import Foundation
import MLX
import MLXNN

// The FLUX.2 autoencoder's decoder, after mflux's `flux2_vae/*`. Activations are kept
// channels-last throughout (MLX's convolution layout); the reference transposes around every
// convolution, which changes nothing numerically. The encoder is not needed for text-to-image
// and its weights are skipped when loading.

final class Flux2ResnetBlock2D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, eps: Float, groups: Int) {
        _norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inChannels, eps: eps, pytorchCompatible: true)
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        _norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outChannels, eps: eps, pytorchCompatible: true)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        _convShortcut.wrappedValue = inChannels != outChannels
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, stride: 1, padding: 0)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x.asType(.float32)).asType(modelPrecision)
        h = conv1(silu(h))
        h = norm2(h.asType(.float32)).asType(modelPrecision)
        h = conv2(silu(h))
        let residual = convShortcut.map { $0(x) } ?? x
        return h + residual
    }
}

final class Flux2AttentionBlock: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(channels: Int, groups: Int, eps: Float) {
        _groupNorm.wrappedValue = GroupNorm(groupCount: groups, dimensions: channels, eps: eps, pytorchCompatible: true)
        _toQ.wrappedValue = Linear(channels, channels)
        _toK.wrappedValue = Linear(channels, channels)
        _toV.wrappedValue = Linear(channels, channels)
        _toOut.wrappedValue = Linear(channels, channels)
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
        return x + toOut(merged)
    }
}

final class Flux2Upsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Nearest-neighbour doubling of height and width.
        let doubled = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
        return conv(doubled)
    }
}

final class Flux2UNetMidBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [Flux2AttentionBlock]

    init(channels: Int, eps: Float, groups: Int) {
        _resnets.wrappedValue = [
            Flux2ResnetBlock2D(inChannels: channels, outChannels: channels, eps: eps, groups: groups),
            Flux2ResnetBlock2D(inChannels: channels, outChannels: channels, eps: eps, groups: groups),
        ]
        _attentions.wrappedValue = [Flux2AttentionBlock(channels: channels, groups: groups, eps: eps)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resnets[0](x)
        h = attentions[0](h)
        return resnets[1](h)
    }
}

final class Flux2UpDecoderBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [Flux2Upsample2D]

    init(inChannels: Int, outChannels: Int, numLayers: Int, eps: Float, groups: Int, addUpsample: Bool) {
        _resnets.wrappedValue = (0 ..< numLayers).map { index in
            Flux2ResnetBlock2D(inChannels: index == 0 ? inChannels : outChannels, outChannels: outChannels, eps: eps, groups: groups)
        }
        _upsamplers.wrappedValue = addUpsample ? [Flux2Upsample2D(channels: outChannels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for resnet in resnets { h = resnet(h) }
        for upsampler in upsamplers { h = upsampler(h) }
        return h
    }
}

final class Flux2Decoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: Flux2UNetMidBlock2D
    @ModuleInfo(key: "up_blocks") var upBlocks: [Flux2UpDecoderBlock2D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(inChannels: Int = 32, outChannels: Int = 3, blockOutChannels: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, groups: Int = 32, eps: Float = 1e-6) {
        _convIn.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: blockOutChannels[blockOutChannels.count - 1], kernelSize: 3, stride: 1, padding: 1)
        _midBlock.wrappedValue = Flux2UNetMidBlock2D(channels: blockOutChannels[blockOutChannels.count - 1], eps: eps, groups: groups)
        let reversedChannels = Array(blockOutChannels.reversed())
        _upBlocks.wrappedValue = reversedChannels.enumerated().map { index, outChannels in
            Flux2UpDecoderBlock2D(
                inChannels: index == 0 ? outChannels : reversedChannels[index - 1],
                outChannels: outChannels,
                numLayers: layersPerBlock + 1,
                eps: eps,
                groups: groups,
                addUpsample: index != reversedChannels.count - 1
            )
        }
        _convNormOut.wrappedValue = GroupNorm(groupCount: groups, dimensions: blockOutChannels[0], eps: eps, pytorchCompatible: true)
        _convOut.wrappedValue = Conv2d(inputChannels: blockOutChannels[0], outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        h = midBlock(h)
        for block in upBlocks { h = block(h) }
        h = convNormOut(h.asType(.float32)).asType(modelPrecision)
        h = silu(h).asType(modelPrecision)
        return convOut(h)
    }
}

/// Running statistics the transformer's packed latents are normalized with.
final class Flux2BatchNormStats: Module {
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float = 1e-4

    init(features: Int) {
        _runningMean.wrappedValue = MLXArray.zeros([features])
        _runningVar.wrappedValue = MLXArray.ones([features])
        super.init()
    }
}

public final class Flux2VAE: Module {
    @ModuleInfo(key: "decoder") var decoder: Flux2Decoder
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats

    static let latentChannels = 32

    public override init() {
        _decoder.wrappedValue = Flux2Decoder()
        _postQuantConv.wrappedValue = Conv2d(inputChannels: Self.latentChannels, outputChannels: Self.latentChannels, kernelSize: 1, stride: 1, padding: 0)
        _bn.wrappedValue = Flux2BatchNormStats(features: 4 * Self.latentChannels)
        super.init()
    }

    /// The encoder and its quantization convolution are not part of this decoder-only port.
    public static func ignoresKey(_ key: String) -> Bool {
        key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.")
    }

    /// From the transformer's packed latents [B, h, w, 128] (channels last) to an image
    /// [B, H, W, 3] in [-1, 1]: de-normalize with the running statistics, unshuffle the 2 × 2
    /// patches, then decode.
    public func decodePacked(_ packed: MLXArray) -> MLXArray {
        let std = sqrt(bn.runningVar + bn.eps)
        let denormalized = packed * std + bn.runningMean
        let unpacked = Self.unpatchify(denormalized)
        return decode(unpacked)
    }

    /// [B, h, w, 128] → [B, 2h, 2w, 32]: channel c·4 + dy·2 + dx lands at pixel (2y + dy, 2x + dx).
    static func unpatchify(_ latents: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
        let c = channels / 4
        // Reference (channels first): reshape (B, C, 2, 2, H, W) → transpose (0, 1, 4, 2, 5, 3) → (B, C, 2H, 2W).
        let x = latents.reshaped([batch, height, width, c, 2, 2])       // b, y, x, c, dy, dx
        return x.transposed(0, 1, 4, 2, 5, 3).reshaped([batch, height * 2, width * 2, c])
    }

    /// [B, H/8, W/8, 32] channels-last latents → image.
    public func decode(_ latents: MLXArray) -> MLXArray {
        decoder(postQuantConv(latents))
    }
}
