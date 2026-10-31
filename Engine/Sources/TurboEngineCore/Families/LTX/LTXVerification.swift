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

    /// The DurationHead's seconds for given connector contexts.
    public func durationSeconds(video: MLXArray, audio: MLXArray) throws -> MLXArray {
        let head = try LTXDurationHead.load(pack.appending(path: "duration_head.safetensors"))
        let seconds = head(video: video, audio: audio)
        eval(seconds)
        return seconds
    }

    public static func durationFrames(seconds: Double, fps: Double, minFrames: Int, maxFrames: Int) -> Int {
        LTXDurationHead.frames(seconds: seconds, fps: fps, minFrames: minFrames, maxFrames: maxFrames)
    }

    /// With `lastFrame` (a pixel frame), a keyframe image's positions follow the clip's.
    public static func positions(
        frames: Int, height: Int, width: Int, fps: Double, audioTokens: Int, lastFrame: Int? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        var video = videoPositions(frames: frames, height: height, width: width, fps: fps)
        if let lastFrame {
            video = concatenated([video, keyframePositions(frame: lastFrame, height: height, width: width, fps: fps)], axis: 1)
        }
        return (video, audioPositions(audioTokens))
    }

    /// A stage's video state as the pipeline builds it: stage 1's noise (`seed`) or stage 2's
    /// renoised upsampled `latent`, the first frame's tokens put in, the last image's appended.
    public static func conditionedState(
        shape: [Int], seed: Int, upsampled latent: MLXArray? = nil, firstFrame: MLXArray?, lastFrame: MLXArray?, tokensPerFrame: Int
    ) -> (latent: MLXArray, clean: MLXArray, mask: MLXArray) {
        var state = latent.map { renoised($0, sigma: LTXConfig.stage2Sigmas[0], seed: seed + LTXConfig.stage2SeedOffset) }
            ?? initialState(shape: shape, seed: seed)
        if let firstFrame { state = conditioned(state, firstFrame: firstFrame, tokensPerFrame: tokensPerFrame) }
        if let lastFrame { state = appendingKeyframe(state, tokens: lastFrame) }
        return (state.latent, state.clean, state.mask)
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

    /// One loop step from its pass's velocities (the reference's own, so that only the update is
    /// checked): x0, the preserved tokens put back, the Euler step or (`ancestral`) the ancestral
    /// one with the given draws (none on the last step). The next video and audio states.
    public static func loopStep(
        video: (latent: MLXArray, clean: MLXArray, mask: MLXArray, velocity: MLXArray),
        audio: (latent: MLXArray, clean: MLXArray, mask: MLXArray, velocity: MLXArray),
        sigma: Double, next: Double, ancestral: Bool, noise: (video: MLXArray, audio: MLXArray)?
    ) -> (video: MLXArray, audio: MLXArray) {
        func state(_ part: (latent: MLXArray, clean: MLXArray, mask: MLXArray, velocity: MLXArray)) -> LTXLatentState {
            LTXLatentState(latent: part.latent, clean: part.clean, mask: part.mask,
                           uniform: part.mask.asType(.float32).min().item(Float.self) == 1)
        }
        let (videoState, audioState) = (state(video), state(audio))
        let pass = passSigmas(sigma, video: videoState)
        let nextVideo = update(video.latent, velocity: video.velocity, sigma: pass.video, state: videoState, from: sigma, to: next,
                               ancestral: ancestral, noise: noise?.video)
        let nextAudio = update(audio.latent, velocity: audio.velocity, sigma: pass.audio, state: audioState, from: sigma, to: next,
                               ancestral: ancestral, noise: noise?.audio)
        eval(nextVideo, nextAudio)
        return (nextVideo, nextAudio)
    }

    /// [1, 128, F, H, W] stage-1 latent → normalized [1, 128, F, 2H, 2W].
    public func upsampleLatent(_ latent: MLXArray) throws -> MLXArray {
        let (f, h, w) = (latent.shape[2], latent.shape[3], latent.shape[4])
        let tokens = latent.transposed(0, 2, 3, 4, 1).reshaped([1, -1, 128])
        return try upsampleForVerification(tokens, frames: f, height: h, width: w)
            .reshaped([1, f, h * 2, w * 2, 128]).transposed(0, 4, 1, 2, 3)
    }

    /// A conditioning picture's pixels at `width` × `height`, as the pipeline prepares them for
    /// the encoder: [1, 3, H, W] bfloat16 in [-1, 1].
    public static func imagePixels(path: String, width: Int, height: Int) throws -> MLXArray {
        try LTXReferenceImage(path: path).pixels(width: width, height: height)
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

/// The full pipeline's stage 1 for `verify` (make_ltx_fixture.py --quality): one guided step's
/// passes on the reference's own inputs, and the whole guided loop, with the dev transformer.
extension LTXVideoModel {
    /// The four passes' x0, keyed `<stream>_<pass>` (cond, uncond, perturbed, isolated).
    public func guidedPasses(
        video: MLXArray, audio: MLXArray, sigma: Double, text: (video: MLXArray, audio: MLXArray),
        negative: (video: MLXArray, audio: MLXArray), guidance: Double, videoPositions: MLXArray, audioPositions: MLXArray,
        keyframeTokens: Int
    ) throws -> [String: MLXArray] {
        let state = LTXLatentState(latent: video, clean: MLXArray.zeros(video.shape, dtype: video.dtype),
                                   mask: MLXArray.ones([video.shape[0], video.shape[1], 1], dtype: .bfloat16), uniform: true)
        let passes = try Self.guidedPasses(
            transformer: try loadedTransformer(.dev), videoX: video, audioX: audio, sigma: sigma, video: state,
            text: text, negative: negative, videoGuider: .video(cfg: guidance), audioGuider: .audio,
            videoPositions: videoPositions, audioPositions: audioPositions, keyframeTokens: keyframeTokens
        )
        var outputs: [String: MLXArray] = ["video_cond": passes.cond.video, "audio_cond": passes.cond.audio]
        for (label, pass) in [("uncond", passes.uncond), ("perturbed", passes.perturbed), ("isolated", passes.isolated)] {
            if let pass {
                outputs["video_\(label)"] = pass.video
                outputs["audio_\(label)"] = pass.audio
            }
        }
        return outputs
    }

    /// The guided loop from given (text-to-video) states.
    public func guidedStageOne(
        video: MLXArray, audio: MLXArray, sigmas: [Double], text: (video: MLXArray, audio: MLXArray),
        negative: (video: MLXArray, audio: MLXArray), guidance: Double, videoPositions: MLXArray, audioPositions: MLXArray,
        keyframeTokens: Int
    ) throws -> (video: MLXArray, audio: MLXArray) {
        func state(_ x: MLXArray) -> LTXLatentState {
            LTXLatentState(latent: x, clean: MLXArray.zeros(x.shape, dtype: x.dtype),
                           mask: MLXArray.ones([x.shape[0], x.shape[1], 1], dtype: .bfloat16), uniform: true)
        }
        return try guidedDenoise(
            transformer: try loadedTransformer(.dev), video: state(video), audio: state(audio), sigmas: sigmas,
            text: text, negative: negative, videoGuider: .video(cfg: guidance), audioGuider: .audio,
            videoPositions: videoPositions, audioPositions: audioPositions, keyframeTokens: keyframeTokens,
            onStep: {}, isCancelled: { false }
        )
    }
}
