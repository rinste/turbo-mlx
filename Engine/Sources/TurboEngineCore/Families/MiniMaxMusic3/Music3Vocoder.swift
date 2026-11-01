import Foundation
import MLX
import MLXNN

// MiniMax Music 3's Flow-VAE decoder (`vocoder.py`), DAC-style: the 128 latent channels as two stereo halves of 64,
// a 1×1 then a 7-wide convolution, four upsampling blocks (8, 8, 4, 2: 512 samples a latent) of a transposed
// convolution and three dilated residual units, snake activations throughout, tanh at the end. The pack stores the
// weight-normed convolutions fused. Activations are channels-second ([N, C, L]) between layers, as in the reference.

/// `Snake1d`: x + sin²(αx) / (α + 1e-9), α per channel.
public final class Music3Snake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.ones([1, channels, 1])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alpha = self.alpha.asType(x.dtype)
        return x + pow(sin(alpha * x), 2) / (alpha + Float(1e-9))
    }
}

public final class Music3ResidualUnit: Module {
    @ModuleInfo(key: "snake1") var snake1: Music3Snake
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "snake2") var snake2: Music3Snake
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(dim: Int, dilation: Int) {
        _snake1.wrappedValue = Music3Snake(channels: dim)
        _conv1.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3 * dilation, dilation: dilation)
        _snake2.wrappedValue = Music3Snake(channels: dim)
        _conv2.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1(snake1(x).transposed(0, 2, 1))
        y = conv2(snake2(y.transposed(0, 2, 1)).transposed(0, 2, 1))
        return x + y.transposed(0, 2, 1)
    }
}

public final class Music3VocoderBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: Music3Snake
    @ModuleInfo(key: "conv_t1") var convT1: ConvTransposed1d
    @ModuleInfo(key: "res_unit1") var resUnit1: Music3ResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: Music3ResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: Music3ResidualUnit

    init(input: Int, output: Int, stride: Int) {
        _snake1.wrappedValue = Music3Snake(channels: input)
        _convT1.wrappedValue = ConvTransposed1d(
            inputChannels: input, outputChannels: output, kernelSize: 2 * stride, stride: stride, padding: (stride + 1) / 2)
        _resUnit1.wrappedValue = Music3ResidualUnit(dim: output, dilation: 1)
        _resUnit2.wrappedValue = Music3ResidualUnit(dim: output, dilation: 3)
        _resUnit3.wrappedValue = Music3ResidualUnit(dim: output, dilation: 9)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = convT1(snake1(x).transposed(0, 2, 1)).transposed(0, 2, 1)
        return resUnit3(resUnit2(resUnit1(y)))
    }
}

public final class Music3Vocoder: Module {
    @ModuleInfo(key: "dec_in_proj") var decInProj: Conv1d
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [Music3VocoderBlock]
    @ModuleInfo(key: "snake_out") var snakeOut: Music3Snake
    @ModuleInfo(key: "conv_out") var convOut: Conv1d
    let latentHalf: Int

    init(config: Music3Config) {
        latentHalf = config.ditInChannels / 2
        let hidden = config.vocoderHiddenDim
        _decInProj.wrappedValue = Conv1d(inputChannels: latentHalf, outputChannels: config.vocoderInputDim, kernelSize: 1)
        _convIn.wrappedValue = Conv1d(inputChannels: config.vocoderInputDim, outputChannels: hidden, kernelSize: 7, padding: 3)
        _blocks.wrappedValue = config.vocoderUpsamplingRatios.enumerated().map { index, stride in
            Music3VocoderBlock(input: hidden >> index, output: hidden >> (index + 1), stride: stride)
        }
        let output = hidden >> config.vocoderUpsamplingRatios.count
        _snakeOut.wrappedValue = Music3Snake(channels: output)
        _convOut.wrappedValue = Conv1d(inputChannels: output, outputChannels: 1, kernelSize: 7, padding: 3)
        super.init()
    }

    /// The waveform [B, 2, L × 512] in −1…1 of `latents` [B, 128, L].
    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let (batch, length) = (latents.dim(0), latents.dim(2))
        var hidden = latents.reshaped(batch * 2, latentHalf, length)
        hidden = decInProj(hidden.transposed(0, 2, 1))
        hidden = convIn(hidden).transposed(0, 2, 1)
        for block in blocks { hidden = block(hidden) }
        let wave = tanh(convOut(snakeOut(hidden).transposed(0, 2, 1))).transposed(0, 2, 1)
        return wave.reshaped(batch, 2, -1)
    }
}
