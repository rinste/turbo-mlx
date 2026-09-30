import Foundation
import MLX
import MLXNN

// The spatial ×2 latent upsampler between the two stages (dgrauet's `LatentUpsampler`, the
// `spatial_upscaler_x2_v1_1` weights, `_v1_0` on LTX-2.5, the same shapes): 3D residual blocks with group norms around a per-frame 2D
// convolution and a pixel shuffle. It works on un-normalized latents; the pipeline denormalizes
// before and normalizes after with the video encoder's statistics.

final class LTXUpsamplerResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv3d
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv3d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm

    init(channels: Int) {
        _conv1.wrappedValue = Conv3d(inputChannels: channels, outputChannels: channels, kernelSize: .init(3), padding: .init(1))
        _norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, pytorchCompatible: true)
        _conv2.wrappedValue = Conv3d(inputChannels: channels, outputChannels: channels, kernelSize: .init(3), padding: .init(1))
        _norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, pytorchCompatible: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = silu(norm1(conv1(x)))
        return silu(norm2(conv2(h)) + x)
    }
}

final class LTXLatentUpsampler: Module {
    @ModuleInfo(key: "initial_conv") var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") var initialNorm: GroupNorm
    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXUpsamplerResBlock]
    @ModuleInfo(key: "upsampler") var upsampler: [Conv2d]
    @ModuleInfo(key: "post_upsample_res_blocks") var postBlocks: [LTXUpsamplerResBlock]
    @ModuleInfo(key: "final_conv") var finalConv: Conv3d

    init(inChannels: Int = 128, midChannels: Int = 1024, blocksPerStage: Int = 4) {
        _initialConv.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: midChannels, kernelSize: .init(3), padding: .init(1))
        _initialNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: midChannels, pytorchCompatible: true)
        _resBlocks.wrappedValue = (0 ..< blocksPerStage).map { _ in LTXUpsamplerResBlock(channels: midChannels) }
        _upsampler.wrappedValue = [Conv2d(inputChannels: midChannels, outputChannels: 4 * midChannels, kernelSize: 3, padding: 1)]
        _postBlocks.wrappedValue = (0 ..< blocksPerStage).map { _ in LTXUpsamplerResBlock(channels: midChannels) }
        _finalConv.wrappedValue = Conv3d(inputChannels: midChannels, outputChannels: inChannels, kernelSize: .init(3), padding: .init(1))
        super.init()
    }

    /// [B, C, F, H, W] → [B, C, F, 2H, 2W].
    func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var x = latent.transposed(0, 2, 3, 4, 1)
        x = silu(initialNorm(initialConv(x)))
        for block in resBlocks { x = block(x) }

        // A 2D convolution on every frame, then a 2 × 2 pixel shuffle, channels outermost.
        let (b, d, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = upsampler[0](x.reshaped([b * d, h, w, c]))
        let channels = x.shape[3] / 4
        x = x.reshaped([b * d, h, w, channels, 2, 2]).transposed(0, 1, 4, 2, 5, 3).reshaped([b, d, h * 2, w * 2, channels])

        for block in postBlocks { x = block(x) }
        x = finalConv(x)
        return x.transposed(0, 4, 1, 2, 3)
    }

    /// The keys under the file's stem (`spatial_upscaler_x2_v1_1` on LTX-2.3, `_v1_0` on 2.5).
    static func weights(_ tensors: [String: MLXArray], stem: String = "spatial_upscaler_x2_v1_1") -> [String: MLXArray] {
        stripping("\(stem).", from: tensors)
    }
}
