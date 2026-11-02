import Foundation
import MLX
import MLXNN

// The decoder of MOSS-SoundEffect's continuous DAC VAE (`DAC.decode` with `continuous=True`, `dac_vae.py` in OpenMOSS's
// MOSS-TTS): the 1×1 `post_quant_conv`, a 7-wide convolution to 2048 channels, five upsampling blocks (8, 5, 4, 3, 2:
// 960 samples a latent) of a snake, a transposed convolution and three dilated residual units, then a snake, a
// 7-wide convolution to one channel and tanh. The MLX pack stores the weight-normed convolutions fused, channels last
// ([O, K, I]), the snakes' α as [1, 1, C], under the reference's tree with `.layers.` for its Sequentials. It runs
// in float32, as the reference decodes. Activations are channels-last ([N, L, C]) throughout.

/// `Snake1d`: x + sin²(αx) / (α + 1e-9), α per channel (`snake` adds 1e-9 to α before the reciprocal).
final class MossSFXSnake: Module, UnaryLayer {
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.ones([1, 1, channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + reciprocal(alpha + Float(1e-9)) * square(sin(alpha * x))
    }
}

/// `ResidualUnit`: snake, 7-wide dilated convolution, snake, 1×1 convolution, added to the input (same length).
final class MossSFXResidualUnit: Module {
    @ModuleInfo(key: "block") var block: MossSFXSequential

    init(dim: Int, dilation: Int) {
        _block.wrappedValue = MossSFXSequential([
            MossSFXSnake(channels: dim),
            Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3 * dilation, dilation: dilation),
            MossSFXSnake(channels: dim),
            Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1),
        ])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + block(x) }
}

/// A `nn.Sequential` saved as `<name>.layers.<i>`: snakes, convolutions, residual units and blocks in order.
final class MossSFXSequential: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(_ layers: [Module]) {
        _layers.wrappedValue = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.reduce(x) { hidden, layer in
            switch layer {
            case let unit as MossSFXResidualUnit: unit(hidden)
            case let block as MossSFXDecoderBlock: block(hidden)
            case let unary as UnaryLayer: unary(hidden)
            default: fatalError("MossSFXSequential: \(type(of: layer)) is not a layer")
            }
        }
    }
}

/// `DecoderBlock`: snake, the transposed convolution that upsamples by `stride` (kernel 2·stride, padding
/// ⌈stride/2⌉, output padding stride mod 2: exactly stride times longer), three residual units dilated 1, 3, 9.
final class MossSFXDecoderBlock: Module {
    @ModuleInfo(key: "block") var block: MossSFXSequential

    init(input: Int, output: Int, stride: Int) {
        _block.wrappedValue = MossSFXSequential([
            MossSFXSnake(channels: input),
            ConvTransposed1d(inputChannels: input, outputChannels: output, kernelSize: 2 * stride, stride: stride,
                             padding: (stride + 1) / 2, outputPadding: stride % 2),
            MossSFXResidualUnit(dim: output, dilation: 1),
            MossSFXResidualUnit(dim: output, dilation: 3),
            MossSFXResidualUnit(dim: output, dilation: 9),
        ])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block(x) }
}

final class MossSFXTanh: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { tanh(x) }
}

/// `Decoder`: the input convolution, the upsampling blocks, the closing snake, convolution and tanh, as `model`.
final class MossSFXDecoder: Module {
    @ModuleInfo(key: "model") var model: MossSFXSequential

    init(config: MossSoundEffectConfig) {
        let channels = config.decoderDim
        var layers: [Module] = [Conv1d(inputChannels: config.latentChannels, outputChannels: channels, kernelSize: 7, padding: 3)]
        for (index, stride) in config.decoderRates.enumerated() {
            layers.append(MossSFXDecoderBlock(input: channels >> index, output: channels >> (index + 1), stride: stride))
        }
        let output = channels >> config.decoderRates.count
        layers += [MossSFXSnake(channels: output), Conv1d(inputChannels: output, outputChannels: 1, kernelSize: 7, padding: 3), MossSFXTanh()]
        _model.wrappedValue = MossSFXSequential(layers)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { model(x) }
}

public final class MossSoundEffectVAE: Module {
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv1d
    @ModuleInfo(key: "decoder") var decoder: MossSFXDecoder

    let config: MossSoundEffectConfig

    public init(config: MossSoundEffectConfig) {
        self.config = config
        _postQuantConv.wrappedValue = Conv1d(inputChannels: config.latentChannels, outputChannels: config.latentChannels, kernelSize: 1)
        _decoder.wrappedValue = MossSFXDecoder(config: config)
        super.init()
    }

    /// The waveform [B, L·960] in −1…1 of latents [B, 128, L], in float32.
    public func decode(_ latents: MLXArray) -> MLXArray {
        let hidden = postQuantConv(latents.asType(.float32).transposed(0, 2, 1))
        return decoder(hidden).squeezed(axis: -1)
    }

    /// The pack's decoder tensors (the encoder and `quant_conv` are not needed to make sound).
    public static func ignoresKey(_ key: String) -> Bool {
        key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.")
    }
}
