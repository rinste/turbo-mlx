import Foundation
import MLX
import MLXNN

// SeedVR2's autoencoder, after mflux's `seedvr2_vae/*`: causal 3D convolutions, 8 × 8 pixels per
// latent, 16 latent channels. Activations are channels-last, [B, T, H, W, C], where mflux keeps
// them channels-first and transposes around each convolution and norm; the operations and their
// dtypes are the same (norms in float32 cast to bfloat16, the checkpoint's float16 weights). The
// original checkpoint stores convolution weights as [O, I, kt, kh, kw]; `SeedVR2VAE.load`
// transposes them as mflux's weight mapping does.

/// `CausalConv3d`: the first frame repeated in front instead of temporal padding, zeros around
/// each frame.
final class SeedVR2CausalConv3d: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray

    let kernel: (t: Int, h: Int, w: Int)
    let stride: (t: Int, h: Int, w: Int)
    let padding: (t: Int, h: Int, w: Int)
    let usePaddingCausal: Bool

    init(_ inChannels: Int, _ outChannels: Int, kernel: (Int, Int, Int) = (3, 3, 3), stride: (Int, Int, Int) = (1, 1, 1),
         padding: (Int, Int, Int) = (1, 1, 1), usePaddingCausal: Bool = false) {
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.usePaddingCausal = usePaddingCausal
        _weight.wrappedValue = MLXArray.zeros([outChannels, kernel.0, kernel.1, kernel.2, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        var temporalPadding = padding.t
        if kernel.t > 1 {
            let causalPad = usePaddingCausal ? 2 * padding.t : kernel.t - 1
            if x.shape[1] == 1 && causalPad == kernel.t - 1 {
                // A still image: every frame the kernel sees is the picture repeated, so this is
                // one 2D convolution with the kernel's frames added up (in float32, the precision
                // the convolution runs in). A third of the work, and none of the buffers of the
                // three per-frame convolutions MLX (0.32) would split the 3D one into.
                return singleFrame(x, weight.asType(.float32).sum(axis: 1))
            }
            if causalPad > 0 {
                x = concatenated([repeated(x[0..., 0 ..< 1], count: causalPad, axis: 1), x], axis: 1)
            }
            temporalPadding = 0
        } else if x.shape[1] == 1 && padding.t == 0 {
            return singleFrame(x, weight[0..., 0])
        }
        let out = convGeneral(x, weight, strides: [stride.t, stride.h, stride.w], padding: [temporalPadding, padding.h, padding.w])
        return out + bias
    }

    /// The convolution of one frame, [B, 1, H, W, C], with a 2D kernel [O, kh, kw, C].
    private func singleFrame(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
        let out = convGeneral(x.squeezed(axis: 1), weight, strides: [stride.h, stride.w], padding: [padding.h, padding.w])
        return expandedDimensions(out + bias, axis: 1)
    }
}

/// A GroupNorm of 32 groups (PyTorch's grouping) in float32, cast to bfloat16 as mflux casts it.
private func groupNormBF16(_ norm: GroupNorm, _ x: MLXArray) -> MLXArray {
    norm(x.asType(.float32)).asType(.bfloat16)
}

final class SeedVR2ResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: SeedVR2CausalConv3d
    @ModuleInfo(key: "conv2") var conv2: SeedVR2CausalConv3d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: SeedVR2CausalConv3d?

    init(_ inChannels: Int, _ outChannels: Int) {
        _norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: inChannels, eps: 1e-6, pytorchCompatible: true)
        _norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels, eps: 1e-6, pytorchCompatible: true)
        _conv1.wrappedValue = SeedVR2CausalConv3d(inChannels, outChannels)
        _conv2.wrappedValue = SeedVR2CausalConv3d(outChannels, outChannels)
        if inChannels != outChannels {
            _convShortcut.wrappedValue = SeedVR2CausalConv3d(inChannels, outChannels, kernel: (1, 1, 1), padding: (0, 0, 0))
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(groupNormBF16(norm1, x)))
        h = conv2(silu(groupNormBF16(norm2, h)))
        return h + (convShortcut.map { $0(x) } ?? x)
    }
}

/// `Attention3D`: one head over each frame's pixels.
final class SeedVR2Attention3D: Module {
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
        let (b, t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        let normed = groupNormBF16(groupNorm, x.reshaped([b * t, h * w, c]))
        let q = expandedDimensions(toQ(normed), axis: 1)
        let k = expandedDimensions(toK(normed), axis: 1)
        let v = expandedDimensions(toV(normed), axis: 1)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: pow(Float(c), -0.5), mask: nil)
        return toOut[0](attended.squeezed(axis: 1)).reshaped([b, t, h, w, c]) + x
    }
}

final class SeedVR2MidBlock: Module {
    @ModuleInfo(key: "attentions") var attentions: [SeedVR2Attention3D]
    @ModuleInfo(key: "resnets") var resnets: [SeedVR2ResnetBlock]

    init(channels: Int) {
        _attentions.wrappedValue = [SeedVR2Attention3D(channels: channels)]
        _resnets.wrappedValue = [SeedVR2ResnetBlock(channels, channels), SeedVR2ResnetBlock(channels, channels)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resnets[1](attentions[0](resnets[0](x)))
    }
}

/// `Downsample3D`: a zero row and column after the picture, then a stride-2 convolution (and
/// stride 2 in time for the temporal ones).
final class SeedVR2Downsample: Module {
    @ModuleInfo(key: "conv") var conv: SeedVR2CausalConv3d

    init(channels: Int, spatialOnly: Bool) {
        let (kt, st, pt) = spatialOnly ? (1, 1, 0) : (3, 2, 1)
        _conv.wrappedValue = SeedVR2CausalConv3d(channels, channels, kernel: (kt, 3, 3), stride: (st, 2, 2), padding: (pt, 0, 0))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [.init(0), .init(0), .init((0, 1)), .init((0, 1)), .init(0)]))
    }
}

final class SeedVR2DownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [SeedVR2ResnetBlock]
    @ModuleInfo(key: "downsamplers") var downsamplers: [SeedVR2Downsample]

    init(_ inChannels: Int, _ outChannels: Int, layers: Int, downsample: Bool, temporalDown: Bool) {
        _resnets.wrappedValue = (0 ..< layers).map { SeedVR2ResnetBlock($0 == 0 ? inChannels : outChannels, outChannels) }
        _downsamplers.wrappedValue = downsample ? [SeedVR2Downsample(channels: outChannels, spatialOnly: !temporalDown)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for resnet in resnets { x = resnet(x) }
        for downsampler in downsamplers { x = downsampler(x) }
        return x
    }
}

/// `Upsample3D`: a 1 × 1 × 1 convolution to 4 (or 8, with time) times the channels, rearranged
/// into 2 × 2 pixels (and 2 frames, of which a single frame keeps the first), then a convolution.
final class SeedVR2Upsample: Module {
    @ModuleInfo(key: "conv") var conv: SeedVR2CausalConv3d
    @ModuleInfo(key: "upscale_conv") var upscaleConv: SeedVR2CausalConv3d

    let temporalFactor: Int

    init(channels: Int, temporalUp: Bool) {
        temporalFactor = temporalUp ? 2 : 1
        _conv.wrappedValue = SeedVR2CausalConv3d(channels, channels, usePaddingCausal: true)
        _upscaleConv.wrappedValue = SeedVR2CausalConv3d(channels, channels * 4 * temporalFactor, kernel: (1, 1, 1), padding: (0, 0, 0))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        let tf = temporalFactor
        // mflux splits the channels as (sf_h, sf_w, tf, C).
        var y = upscaleConv(x).reshaped([b, t, h, w, 2, 2, tf, c]).transposed(0, 1, 6, 2, 4, 3, 5, 7)
            .reshaped([b, t * tf, h * 2, w * 2, c])
        if t == 1 && tf > 1 { y = y[0..., 0 ..< 1] }
        return conv(y)
    }
}

final class SeedVR2UpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [SeedVR2ResnetBlock]
    @ModuleInfo(key: "upsamplers") var upsamplers: [SeedVR2Upsample]

    init(_ inChannels: Int, _ outChannels: Int, layers: Int, upsample: Bool, temporalUp: Bool) {
        _resnets.wrappedValue = (0 ..< layers).map { SeedVR2ResnetBlock($0 == 0 ? inChannels : outChannels, outChannels) }
        _upsamplers.wrappedValue = upsample ? [SeedVR2Upsample(channels: outChannels, temporalUp: temporalUp)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for resnet in resnets { x = resnet(x) }
        for upsampler in upsamplers { x = upsampler(x) }
        return x
    }
}

final class SeedVR2Encoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: SeedVR2CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [SeedVR2DownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: SeedVR2MidBlock
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: SeedVR2CausalConv3d

    init(channels: [Int], latentChannels: Int, layers: Int = 2, temporalDownBlocks: Int = 2) {
        _convIn.wrappedValue = SeedVR2CausalConv3d(3, channels[0])
        var output = channels[0]
        _downBlocks.wrappedValue = channels.enumerated().map { i, channel in
            let input = output
            output = channel
            let isFinal = i == channels.count - 1
            let temporalDown = i >= channels.count - temporalDownBlocks - 1 && !isFinal
            return SeedVR2DownBlock(input, output, layers: layers, downsample: !isFinal, temporalDown: temporalDown)
        }
        _midBlock.wrappedValue = SeedVR2MidBlock(channels: channels.last!)
        _convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels.last!, eps: 1e-6, pytorchCompatible: true)
        _convOut.wrappedValue = SeedVR2CausalConv3d(channels.last!, 2 * latentChannels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)
        for block in downBlocks { x = block(x) }
        x = midBlock(x)
        return convOut(silu(groupNormBF16(convNormOut, x)))
    }
}

final class SeedVR2Decoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: SeedVR2CausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: SeedVR2MidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [SeedVR2UpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: SeedVR2CausalConv3d

    init(channels: [Int], latentChannels: Int, layers: Int = 3, temporalUpBlocks: Int = 2) {
        let reversed = Array(channels.reversed())
        _convIn.wrappedValue = SeedVR2CausalConv3d(latentChannels, reversed[0])
        _midBlock.wrappedValue = SeedVR2MidBlock(channels: reversed[0])
        var output = reversed[0]
        _upBlocks.wrappedValue = reversed.enumerated().map { i, channel in
            let input = output
            output = channel
            return SeedVR2UpBlock(input, output, layers: layers, upsample: i != reversed.count - 1, temporalUp: i < temporalUpBlocks)
        }
        _convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: reversed.last!, eps: 1e-6, pytorchCompatible: true)
        _convOut.wrappedValue = SeedVR2CausalConv3d(reversed.last!, 3)
        super.init()
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var x = midBlock(convIn(z))
        for block in upBlocks { x = block(x) }
        return convOut(silu(groupNormBF16(convNormOut, x)))
    }
}

public final class SeedVR2VAE: Module {
    @ModuleInfo(key: "encoder") var encoder: SeedVR2Encoder
    @ModuleInfo(key: "decoder") var decoder: SeedVR2Decoder

    let scalingFactor: Float
    let latentChannels: Int

    public init(config: SeedVR2Config) {
        scalingFactor = config.scalingFactor
        latentChannels = config.latentChannels
        _encoder.wrappedValue = SeedVR2Encoder(channels: config.blockOutChannels, latentChannels: config.latentChannels)
        _decoder.wrappedValue = SeedVR2Decoder(channels: config.blockOutChannels, latentChannels: config.latentChannels)
        super.init()
    }

    /// The original checkpoint's tensors as this module names them: convolution weights from
    /// [O, I, kt, kh, kw] to channels-last [O, kt, kh, kw, I].
    static func channelsLast(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        tensors.mapValues { $0.ndim == 5 ? $0.transposed(0, 2, 3, 4, 1) : $0 }
    }

    /// Pixels [1, H, W, 3] in [-1, 1] → the scaled mean of the latent [1, H/8, W/8, 16].
    public func encode(_ x: MLXArray) -> MLXArray {
        let h = encoder(expandedDimensions(x, axis: 1))
        return h[0..., 0, 0..., 0..., 0 ..< latentChannels] * scalingFactor
    }

    /// A latent [1, h, w, 16] → pixels [1, 8h, 8w, 3], roughly in [-1, 1].
    public func decode(_ z: MLXArray) -> MLXArray {
        decoder(expandedDimensions(z / scalingFactor, axis: 1))[0..., 0]
    }
}
