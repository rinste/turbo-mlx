import Foundation
import MLX
import MLXNN

// LTX-2's sound: the audio latents become a stereo mel spectrogram (`AudioVAEDecoder`), a
// BigVGAN v2 vocoder turns it into 16 kHz audio, and a bandwidth extension (a second BigVGAN on
// the mel of that audio, added to a 3× sinc resampling) brings it to 48 kHz. After dgrauet's
// `model/audio_vae/` (`audio_vae.py`, `vocoder.py`, `bwe.py`). The vocoder runs in float32, as
// the reference upcasts it: over its hundred convolutions bfloat16 audibly degrades the sound.

// MARK: - Audio VAE decoder

private func audioPixelNorm(_ x: MLXArray) -> MLXArray { unweightedRMSNorm(x, eps: 1e-6) }

/// `WrappedConv2d`: causal along height (time), padded on top only.
final class LTXAudioConv2d: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    let causalPadding: Int
    let widthPadding: Int

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, padding: Int, causal: Bool) {
        if causal && kernel > 1 {
            causalPadding = kernel - 1
            widthPadding = padding
            _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: .init(kernel), padding: .init(0))
        } else {
            causalPadding = 0
            widthPadding = 0
            _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: .init(kernel), padding: .init(padding))
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard causalPadding > 0 || widthPadding > 0 else { return conv(x) }
        let padded = MLX.padded(x, widths: [.init(0), .init((causalPadding, 0)), .init((widthPadding, widthPadding)), .init(0)])
        return conv(padded)
    }
}

final class LTXAudioResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: LTXAudioConv2d
    @ModuleInfo(key: "conv2") var conv2: LTXAudioConv2d
    @ModuleInfo(key: "nin_shortcut") var ninShortcut: LTXAudioConv2d?

    init(_ inChannels: Int, _ outChannels: Int) {
        _conv1.wrappedValue = LTXAudioConv2d(inChannels, outChannels, kernel: 3, padding: 1, causal: true)
        _conv2.wrappedValue = LTXAudioConv2d(outChannels, outChannels, kernel: 3, padding: 1, causal: true)
        _ninShortcut.wrappedValue = inChannels != outChannels
            ? LTXAudioConv2d(inChannels, outChannels, kernel: 1, padding: 0, causal: false)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(audioPixelNorm(x)))
        h = conv2(silu(audioPixelNorm(h)))
        return h + (ninShortcut.map { $0(x) } ?? x)
    }
}

final class LTXAudioUpsample: Module {
    @ModuleInfo(key: "conv") var conv: LTXAudioConv2d

    init(channels: Int) {
        _conv.wrappedValue = LTXAudioConv2d(channels, channels, kernel: 3, padding: 1, causal: true)
        super.init()
    }

    /// Nearest ×2 on time and frequency, the convolution, then the first row dropped (causal).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let doubled = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
        return conv(doubled)[0..., 1...]
    }
}

final class LTXAudioUpBlock: Module {
    @ModuleInfo(key: "block") var block: [LTXAudioResBlock]
    @ModuleInfo(key: "upsample") var upsample: LTXAudioUpsample?

    init(_ inChannels: Int, _ outChannels: Int, upsample: Bool) {
        _block.wrappedValue = (0 ..< 3).map { LTXAudioResBlock($0 == 0 ? inChannels : outChannels, outChannels) }
        _upsample.wrappedValue = upsample ? LTXAudioUpsample(channels: outChannels) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for resBlock in block { x = resBlock(x) }
        return upsample.map { $0(x) } ?? x
    }
}

final class LTXAudioMidBlock: Module {
    @ModuleInfo(key: "block_1") var block1: LTXAudioResBlock
    @ModuleInfo(key: "block_2") var block2: LTXAudioResBlock

    init(channels: Int) {
        _block1.wrappedValue = LTXAudioResBlock(channels, channels)
        _block2.wrappedValue = LTXAudioResBlock(channels, channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block2(block1(x)) }
}

final class LTXAudioDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: LTXAudioConv2d
    @ModuleInfo(key: "mid") var mid: LTXAudioMidBlock
    @ModuleInfo(key: "up") var up: [LTXAudioUpBlock]
    @ModuleInfo(key: "conv_out") var convOut: LTXAudioConv2d
    @ModuleInfo(key: "per_channel_statistics") var statistics: LTXEncoderStatistics

    override init() {
        _convIn.wrappedValue = LTXAudioConv2d(8, 512, kernel: 3, padding: 1, causal: true)
        _mid.wrappedValue = LTXAudioMidBlock(channels: 512)
        _up.wrappedValue = [
            LTXAudioUpBlock(256, 128, upsample: false),
            LTXAudioUpBlock(512, 256, upsample: true),
            LTXAudioUpBlock(512, 512, upsample: true),
        ]
        _convOut.wrappedValue = LTXAudioConv2d(128, 2, kernel: 3, padding: 1, causal: true)
        _statistics.wrappedValue = LTXEncoderStatistics(channels: 128)
        super.init()
    }

    /// Latent [B, 8, T, 16] → stereo mel [B, 2, 4T − 3, 64]: denormalized over the 128 flattened
    /// channels, then decoded as an image with time as height and frequency as width.
    func decode(_ latent: MLXArray) -> MLXArray {
        let (b, c1, t, c2) = (latent.shape[0], latent.shape[1], latent.shape[2], latent.shape[3])
        var flat = latent.transposed(0, 2, 1, 3).reshaped([b, t, c1 * c2])
        flat = statistics.denormalize(flat)
        var x = flat.reshaped([b, t, c1, c2]).transposed(0, 1, 3, 2)
        x = mid(convIn(x))
        for index in up.indices.reversed() { x = up[index](x) }
        x = convOut(silu(audioPixelNorm(x)))
        return x.transposed(0, 3, 1, 2)
    }

    /// `audio_vae.decoder.*` plus the root statistics, remapped.
    static func weights(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        var weights = stripping("audio_vae.decoder.", from: tensors)
        for (key, value) in stripping("audio_vae.", from: tensors) where key.hasPrefix("per_channel_statistics.") {
            weights[key] = value
        }
        return remappingStatistics(weights)
    }
}

// MARK: - Vocoder

/// `SnakeBeta`: x + sin²(αx)/β with α and β stored as logarithms.
final class LTXSnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.zeros([channels])
        _beta.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha).reshaped([1, 1, -1])
        let b = exp(beta).reshaped([1, 1, -1])
        return x + (1 / (b + 1e-9)) * pow(sin(a * x), 2)
    }
}

final class LTXLowPass: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray

    init(kernel: Int = 12) {
        _filter.wrappedValue = MLXArray.ones([1, kernel, 1])
        super.init()
    }
}

/// [B, T, C] as [B·C, T, 1], the layout the depthwise filters run in, and back.
private func perChannel(_ x: MLXArray) -> MLXArray {
    let (b, t, c) = (x.shape[0], x.shape[1], x.shape[2])
    return x.transposed(0, 2, 1).reshaped([b * c, t, 1])
}

private func fromPerChannel(_ x: MLXArray, batch: Int, channels: Int) -> MLXArray {
    x.reshaped([batch, channels, x.shape[1]]).transposed(0, 2, 1)
}

/// `Activation1d`: 2× upsampling (zeros inserted, a low-pass filter), SnakeBeta, 2× downsampling
/// (low-pass, stride 2), with replicated edges.
final class LTXActivation1d: Module {
    @ModuleInfo(key: "act") var act: LTXSnakeBeta
    /// `upsample.filter`: the upsampler holds just its filter.
    @ModuleInfo(key: "upsample") var upsampleFilter: LTXLowPass
    @ModuleInfo(key: "downsample") var downsample: LTXDownsample1d

    init(channels: Int) {
        _act.wrappedValue = LTXSnakeBeta(channels: channels)
        _upsampleFilter.wrappedValue = LTXLowPass()
        _downsample.wrappedValue = LTXDownsample1d()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(act(upsample(x)))
    }

    private func upsample(_ x: MLXArray) -> MLXArray {
        let (b, t, c) = (x.shape[0], x.shape[1], x.shape[2])
        let filter = upsampleFilter.filter
        // Samples at even positions, zeros between: [B, T, 1, C] with a zero row, flattened.
        let doubled = concatenated([x.expandedDimensions(axis: 2).asType(.float32), MLXArray.zeros([b, t, 1, c])], axis: 2)
            .reshaped([b, t * 2, c])
        var y = perChannel(doubled)
        let k = filter.shape[1]
        let left = repeated(y[0..., 0 ..< 1, 0...], count: k / 2, axis: 1)
        let right = repeated(y[0..., (t * 2 - 1) ..< (t * 2), 0...], count: k / 2 - 1, axis: 1)
        y = conv1d(concatenated([left, y, right], axis: 1), filter)
        return fromPerChannel(y, batch: b, channels: c) * 2
    }
}

final class LTXDownsample1d: Module {
    @ModuleInfo(key: "lowpass") var lowpass: LTXLowPass

    override init() {
        _lowpass.wrappedValue = LTXLowPass()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, c) = (x.shape[0], x.shape[1], x.shape[2])
        var y = perChannel(x)
        let k = lowpass.filter.shape[1]
        let even = k % 2 == 0 ? 1 : 0
        let left = repeated(y[0..., 0 ..< 1, 0...], count: k / 2 - even, axis: 1)
        let right = repeated(y[0..., (t - 1) ..< t, 0...], count: k / 2, axis: 1)
        y = conv1d(concatenated([left, y, right], axis: 1), lowpass.filter, stride: 2)
        return fromPerChannel(y, batch: b, channels: c)
    }
}

/// `AMPBlock1`: three dilated convolutions, each between anti-aliased activations, as residuals.
final class LTXAMPBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "acts1") var acts1: [LTXActivation1d]
    @ModuleInfo(key: "acts2") var acts2: [LTXActivation1d]

    init(channels: Int, kernel: Int, dilations: [Int] = [1, 3, 5]) {
        _convs1.wrappedValue = dilations.map { d in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel * d - d) / 2, dilation: d)
        }
        _convs2.wrappedValue = dilations.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: kernel / 2)
        }
        _acts1.wrappedValue = dilations.map { _ in LTXActivation1d(channels: channels) }
        _acts2.wrappedValue = dilations.map { _ in LTXActivation1d(channels: channels) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for i in convs1.indices {
            let residual = x
            x = convs2[i](acts2[i](convs1[i](acts1[i](x))))
            x = x + residual
        }
        return x
    }
}

/// BigVGAN: `conv_pre`, per stage a transposed convolution then the mean of three AMP blocks,
/// `act_post`, `conv_post`.
final class LTXBigVGAN: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resblocks: [LTXAMPBlock]
    @ModuleInfo(key: "act_post") var actPost: LTXActivation1d
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    let finalTanh: Bool
    private static let kernels = [3, 7, 11]

    init(initialChannels: Int, rates: [Int], kernels: [Int], finalTanh: Bool) {
        self.finalTanh = finalTanh
        _convPre.wrappedValue = Conv1d(inputChannels: 128, outputChannels: initialChannels, kernelSize: 7, padding: 3)
        var ups: [ConvTransposed1d] = []
        var blocks: [LTXAMPBlock] = []
        var channels = initialChannels
        for (rate, kernel) in zip(rates, kernels) {
            ups.append(ConvTransposed1d(inputChannels: channels, outputChannels: channels / 2, kernelSize: kernel, stride: rate,
                                        padding: (kernel - rate) / 2))
            channels /= 2
            for size in Self.kernels { blocks.append(LTXAMPBlock(channels: channels, kernel: size)) }
        }
        _ups.wrappedValue = ups
        _resblocks.wrappedValue = blocks
        _actPost.wrappedValue = LTXActivation1d(channels: channels)
        _convPost.wrappedValue = Conv1d(inputChannels: channels, outputChannels: 2, kernelSize: 7, padding: 3, bias: false)
        super.init()
    }

    /// Mel [B, T, 128] → waveform [B, T·∏rates, 2].
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel)
        for (stage, up) in ups.enumerated() {
            x = up(x)
            var sum = resblocks[stage * 3](x)
            for j in 1 ..< 3 { sum = sum + resblocks[stage * 3 + j](x) }
            x = sum / 3
        }
        x = convPost(actPost(x))
        return finalTanh ? tanh(x) : x
    }
}

final class LTXSTFTBasis: Module {
    @ParameterInfo(key: "forward_basis") var forwardBasis: MLXArray
    @ParameterInfo(key: "inverse_basis") var inverseBasis: MLXArray

    init(fft: Int = 512) {
        _forwardBasis.wrappedValue = MLXArray.zeros([fft + 2, fft, 1])
        _inverseBasis.wrappedValue = MLXArray.zeros([fft + 2, fft, 1])
        super.init()
    }
}

/// `MelSTFT`: a causal STFT by convolution with the stored basis (hop 80), magnitudes, the mel
/// filter bank, log.
final class LTXMelSTFT: Module {
    @ParameterInfo(key: "mel_basis") var melBasis: MLXArray
    @ModuleInfo(key: "stft_fn") var stft: LTXSTFTBasis

    let fft = 512
    let hop = 80

    override init() {
        _melBasis.wrappedValue = MLXArray.zeros([64, 257])
        _stft.wrappedValue = LTXSTFTBasis()
        super.init()
    }

    /// [B, T] → [B, frames, 64].
    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        let x = MLX.padded(waveform.expandedDimensions(axis: -1), widths: [.init(0), .init((fft - hop, 0)), .init(0)])
        let spectrum = conv1d(x, stft.forwardBasis, stride: hop)
        let bins = fft / 2 + 1
        let real = spectrum[0..., 0..., 0 ..< bins]
        let imag = spectrum[0..., 0..., bins...]
        let magnitude = sqrt(real * real + imag * imag + 1e-9)
        return log(maximum(matmul(magnitude, melBasis.transposed()), MLXArray(Float(1e-5))))
    }
}

/// `VocoderWithBWE`: mel (stereo, the two channels' 64 bins side by side) → 48 kHz stereo.
final class LTXVocoder: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resblocks: [LTXAMPBlock]
    @ModuleInfo(key: "act_post") var actPost: LTXActivation1d
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    @ModuleInfo(key: "bwe_generator") var bweGenerator: LTXBigVGAN
    @ModuleInfo(key: "mel_stft") var melSTFT: LTXMelSTFT

    private let resampler = LTXSincResampler(factor: 3)

    override init() {
        // The base vocoder's layers sit at the top of the file, beside the BWE generator's.
        let base = LTXBigVGAN(initialChannels: 1536, rates: [5, 2, 2, 2, 2, 2], kernels: [11, 4, 4, 4, 4, 4], finalTanh: true)
        _convPre.wrappedValue = base.convPre
        _ups.wrappedValue = base.ups
        _resblocks.wrappedValue = base.resblocks
        _actPost.wrappedValue = base.actPost
        _convPost.wrappedValue = base.convPost
        _bweGenerator.wrappedValue = LTXBigVGAN(initialChannels: 512, rates: [6, 5, 2, 2, 2], kernels: [12, 11, 4, 4, 4], finalTanh: false)
        _melSTFT.wrappedValue = LTXMelSTFT()
        super.init()
    }

    private func baseVocoder(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel)
        for (stage, up) in ups.enumerated() {
            x = up(x)
            var sum = resblocks[stage * 3](x)
            for j in 1 ..< 3 { sum = sum + resblocks[stage * 3 + j](x) }
            x = sum / 3
        }
        return tanh(convPost(actPost(x)))
    }

    /// [B, 2, T, 64] mel → [B, 2, samples] at 48 kHz, in the mel's type (computed in float32).
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        let inputType = mel.dtype
        let mel = mel.asType(.float32)
        let (b, c, t, m) = (mel.shape[0], mel.shape[1], mel.shape[2], mel.shape[3])
        let concatenatedMel = mel.transposed(0, 1, 3, 2).reshaped([b, c * m, t]).transposed(0, 2, 1)
        var waveform = baseVocoder(concatenatedMel).transposed(0, 2, 1)

        let length = waveform.shape[2]
        let outputLength = length * 3
        let remainder = length % melSTFT.hop
        if remainder != 0 {
            waveform = MLX.padded(waveform, widths: [.init(0), .init(0), .init((0, melSTFT.hop - remainder))])
        }

        let bweMel = melSTFT(waveform.reshaped([b * c, -1]))
        let frames = bweMel.shape[1]
        let bweInput = bweMel.reshaped([b, c, frames, m]).transposed(0, 1, 3, 2).reshaped([b, c * m, frames]).transposed(0, 2, 1)
        let residual = bweGenerator(bweInput).transposed(0, 2, 1)

        let skip = stacked((0 ..< c).map { resampler(waveform[0..., $0, 0...]) }, axis: 1)
        let common = min(skip.shape[2], residual.shape[2])
        let output = clip(skip[0..., 0..., 0 ..< common] + residual[0..., 0..., 0 ..< common], min: -1, max: 1)
        return output[0..., 0..., 0 ..< min(outputLength, common)].asType(inputType)
    }

    /// `vocoder.` stripped; everything as float32.
    static func weights(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
        stripping("vocoder.", from: tensors).mapValues { $0.asType(.float32) }
    }
}

/// `HannSincResampler`: 3× upsampling with a Hann-windowed sinc (no learned weights).
struct LTXSincResampler {
    let factor: Int
    let kernel: MLXArray
    private let width: Int

    init(factor: Int) {
        self.factor = factor
        let rolloff = 0.99
        let lowpassWidth = 6.0
        let width = Int((lowpassWidth / rolloff).rounded(.up))
        self.width = width
        let size = 2 * width * factor + 1
        let values: [Float] = (0 ..< size).map { index in
            let time = (Double(index) / Double(factor) - Double(width)) * rolloff
            let clamped = min(max(time, -lowpassWidth), lowpassWidth)
            let window = pow(cos(clamped * Double.pi / lowpassWidth / 2), 2)
            let sinc = time == 0 ? 1 : sin(Double.pi * time) / (Double.pi * time)
            return Float(sinc * window * rolloff / Double(factor))
        }
        kernel = MLXArray(values).reshaped([1, size, 1])
    }

    /// [B, T] → [B, 3T]: edges replicated, zeros inserted, a full convolution, scaled, trimmed.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.shape[0], x.shape[1])
        let padded = concatenated([repeated(x[0..., 0 ..< 1], count: width, axis: 1), x,
                                   repeated(x[0..., (t - 1) ..< t], count: width, axis: 1)], axis: 1)
        let paddedLength = padded.shape[1]
        // Zero insertion: each sample followed by factor − 1 zeros, the trailing zeros dropped.
        let spread = concatenated([padded.expandedDimensions(axis: 2),
                                   MLXArray.zeros([b, paddedLength, factor - 1], dtype: padded.dtype)], axis: 2)
            .reshaped([b, paddedLength * factor])[0..., 0 ..< ((paddedLength - 1) * factor + 1)]
        let size = kernel.shape[1]
        let full = MLX.padded(spread.expandedDimensions(axis: -1), widths: [.init(0), .init((size - 1, size - 1)), .init(0)])
        let result = conv1d(full, kernel).squeezed(axis: -1) * Float(factor)
        let padLeft = 2 * width * factor
        let padRight = size - factor
        let trimmed = result[0..., padLeft ..< (result.shape[1] - padRight)]
        return trimmed[0..., 0 ..< (t * factor)]
    }
}
