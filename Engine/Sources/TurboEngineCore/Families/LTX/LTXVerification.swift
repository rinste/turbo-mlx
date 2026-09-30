import Foundation
import MLX

/// The pipeline's stages one at a time, for `turbo-engine verify` to feed the reference's own
/// inputs to each and compare what comes out (Engine/Fixtures/make_ltx_fixture.py).
extension LTXVideoModel {
    public func tokenize(_ prompt: String) throws -> (tokens: MLXArray, mask: MLXArray) {
        guard let prompter = prompterForVerification else { throw Qwen3Prompter.PromptError.noTokenizer(textEncoderFolder) }
        return prompter.tokenize(prompt)
    }

    /// Gemma's 49 hidden states.
    public func gemmaStates(tokens: MLXArray, mask: MLXArray) throws -> [MLXArray] {
        try textModels().gemma.allHiddenStates(tokens: tokens, attentionMask: mask)
    }

    /// The connector's video and audio contexts from Gemma's states; `float32` upcasts the
    /// connector for good (the states are the caller's).
    public func connect(states: [MLXArray], mask: MLXArray, float32: Bool = false) throws -> (video: MLXArray, audio: MLXArray) {
        let connector = try textModels().connector
        if float32 {
            connector.update(parameters: connector.parameters().mapValues { $0.asType(.float32) })
        }
        let result = connector(hiddenStates: states, attentionMask: mask)
        eval(result.video, result.audio)
        return result
    }

    public static func stageOneNoise(shape: [Int], seed: Int) -> MLXArray {
        initialState(shape: shape, seed: seed).latent
    }

    public static func stageTwoVideoNoise(_ latent: MLXArray, seed: Int) -> MLXArray {
        renoised(latent, sigma: LTXConfig.stage2Sigmas[0], seed: seed + LTXConfig.stage2SeedOffset).latent
    }

    public static func stageTwoAudioNoise(_ latent: MLXArray, seed: Int) -> MLXArray {
        renoisedMasked(latent, sigma: LTXConfig.stage2Sigmas[0], seed: seed + LTXConfig.stage2SeedOffset).latent
    }

    /// The ancestral sampler's draws from `seed` (the loop's own, offset already added), in order.
    public static func ancestralNoise(shapes: [[Int]], seed: Int) -> [MLXArray] {
        var sequence = LTXNoiseSequence(seed: seed)
        return shapes.map { sequence.normal($0) }
    }

    public static func positions(frames: Int, height: Int, width: Int, fps: Double, audioTokens: Int) -> (video: MLXArray, audio: MLXArray) {
        (videoPositions(frames: frames, height: height, width: width, fps: fps), audioPositions(audioTokens))
    }

    /// One forward pass: the velocities. `keyframeTokens`: the first latent frame's tokens, which
    /// LTX-2.5 marks.
    public func transformerPass(
        video: MLXArray, audio: MLXArray, sigma: MLXArray, videoTimesteps: MLXArray?,
        videoText: MLXArray, audioText: MLXArray, videoPositions: MLXArray, audioPositions: MLXArray, keyframeTokens: Int = 0
    ) throws -> (video: MLXArray, audio: MLXArray) {
        let result = try transformerForVerification()(
            video: video, audio: audio, sigma: sigma, videoTimesteps: videoTimesteps,
            videoText: videoText, audioText: audioText, videoPositions: videoPositions, audioPositions: audioPositions,
            keyframeTokens: keyframeTokens
        )
        eval(result.video, result.audio)
        return result
    }

    /// A whole stage's loop from given states; with `ancestralSeed`, the ancestral one.
    public func denoiseStage(
        video: (latent: MLXArray, clean: MLXArray, mask: MLXArray), audio: (latent: MLXArray, clean: MLXArray, mask: MLXArray),
        sigmas: [Double], videoText: MLXArray, audioText: MLXArray, videoPositions: MLXArray, audioPositions: MLXArray,
        keyframeTokens: Int = 0, ancestralSeed: Int? = nil
    ) throws -> (video: MLXArray, audio: MLXArray) {
        let videoUniform = video.mask.asType(.float32).min().item(Float.self) == 1
        let audioUniform = audio.mask.asType(.float32).min().item(Float.self) == 1
        return try denoiseForVerification(
            video: LTXLatentState(latent: video.latent, clean: video.clean, mask: video.mask, uniform: videoUniform),
            audio: LTXLatentState(latent: audio.latent, clean: audio.clean, mask: audio.mask, uniform: audioUniform),
            sigmas: sigmas, videoText: videoText, audioText: audioText, videoPositions: videoPositions, audioPositions: audioPositions,
            keyframeTokens: keyframeTokens, ancestralSeed: ancestralSeed
        )
    }

    /// [1, 128, F, H, W] stage-1 latent → normalized [1, 128, F, 2H, 2W].
    public func upsampleLatent(_ latent: MLXArray) throws -> MLXArray {
        let (f, h, w) = (latent.shape[2], latent.shape[3], latent.shape[4])
        let tokens = latent.transposed(0, 2, 3, 4, 1).reshaped([1, -1, 128])
        return try upsampleForVerification(tokens, frames: f, height: h, width: w)
            .reshaped([1, f, h * 2, w * 2, 128]).transposed(0, 4, 1, 2, 3)
    }

    /// The first frame's tokens for given pixels [1, 3, H, W] in [-1, 1]; `float32` upcasts the
    /// encoder and the pixels.
    public func imageTokens(pixels: MLXArray, float32: Bool = false) throws -> MLXArray {
        let encoder = try encoderForVerification()
        if float32 {
            encoder.update(parameters: encoder.parameters().mapValues { $0.asType(.float32) })
        }
        let (h, w) = (pixels.shape[2], pixels.shape[3])
        let input = float32 ? pixels.asType(.float32) : pixels
        let latent = encoder.encode(input.reshaped([1, 3, 1, h, w]))
        eval(latent)
        return latent.transposed(0, 2, 3, 4, 1).reshaped([1, -1, 128])
    }

    /// `float32`: the weights and the latent upcast, to compare the math without bfloat16 rounding.
    public func decodePixels(_ latent: MLXArray, float32: Bool = false) throws -> MLXArray {
        let decoder = try videoDecoderForVerification()
        if float32 {
            decoder.update(parameters: decoder.parameters().mapValues { $0.asType(.float32) })
        }
        let pixels = decoder.decode(float32 ? latent.asType(.float32) : latent)
        eval(pixels)
        return pixels
    }

    /// The decoder's intermediate activations, named as decode_stages.safetensors names them.
    public func decoderStages(_ latent: MLXArray) throws -> [(String, MLXArray)] {
        try videoDecoderForVerification().stages(latent)
    }

    public func decodeMel(_ latent: MLXArray) throws -> MLXArray {
        let mel = try audioDecoderForVerification().decode(latent)
        eval(mel)
        return mel
    }

    public func vocode(_ mel: MLXArray) throws -> MLXArray {
        let waveform = try vocoderForVerification()(mel)
        eval(waveform)
        return waveform
    }
}
