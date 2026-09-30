import Foundation
import MLX
import TurboEngineCore

/// The LTX-2.3 / LTX-2.5 port against dgrauet's ltx-2-mlx on a real pack
/// (Engine/Fixtures/make_ltx_fixture.py): each stage fed the reference's own inputs, so an error
/// shows where it starts. The tokenizer, Gemma's states, the connector, the noise and positions,
/// the first transformer pass of each stage, both loops (2.5's stage 1 ancestral, with its noise),
/// the upsampler, the image encoder (image-to-video), the video decoder, the audio decoder and the
/// vocoder.
enum VerifyLTX {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        // 2.5 fixtures have no Gemma folder: the pack carries its text encoder.
        guard let pack = json["pack"] as? String else {
            print("fixture.json has no pack path"); return false
        }
        let gemma = json["gemma"] as? String
        let prompt = json["prompt"] as? String ?? ""
        let seed = (json["seed"] as? NSNumber)?.intValue ?? 42
        let fps = (json["fps"] as? NSNumber)?.doubleValue ?? 24
        print("loading \(pack)…")
        let model = try LTXVideoModel(pack: URL(fileURLWithPath: pack), textEncoder: gemma.map { URL(fileURLWithPath: $0) })
        let ancestral = model.config.isLTX25
        var ok = true

        // 1. Tokens: the same ids, exactly.
        let (tokens, mask) = try model.tokenize(prompt)
        let wantedTokens = try reference("token_ids")
        let sameTokens = (tokens .== wantedTokens).all().item(Bool.self)
        print("  \(sameTokens ? "✓" : "✗") tokens             \(mask.sum().item(Int.self)) real tokens\(sameTokens ? "" : ", DIFFERENT ids")")
        ok = sameTokens && ok

        // 2. Gemma's states (bfloat16 through 48 layers: rounding grows with depth).
        let states = try model.gemmaStates(tokens: wantedTokens, mask: try reference("attention_mask"))
        for layer in [0, 1, 24, 48] {
            ok = Verify.report("gemma state \(layer)", got: states[layer], want: try reference("gemma_state_\(layer)"), counts: layer < 24) && ok
        }

        // 3. The connector from our states: Gemma's bfloat16 drift carried on (shown, "·"). The
        // check that decides feeds it the reference's own 49 states, below.
        let (videoText, audioText) = try model.connect(states: states, mask: try reference("attention_mask"))
        ok = Verify.report("video context", got: videoText, want: try reference("video_embeds"), counts: false) && ok
        ok = Verify.report("audio context", got: audioText, want: try reference("audio_embeds"), counts: false) && ok
        // The connector alone, from the reference's 49 states: its own math, without Gemma's drift.
        // In bfloat16 LTX-2.5's connector moves by about 1 in 20 with the rounding (shown when a
        // float32 reference exists); in float32 the math decides.
        if let all = try? loadArrays(url: fixture.appending(path: "gemma_states.safetensors")) {
            let referenceStates = (0 ..< 49).compactMap { all["state_\($0)"] }
            let float32 = try? loadArrays(url: fixture.appending(path: "connector_f32.safetensors"))
            let (video, audio) = try model.connect(states: referenceStates, mask: try reference("attention_mask"))
            ok = Verify.report("connector video", got: video, want: try reference("video_embeds"), counts: float32 == nil) && ok
            ok = Verify.report("connector audio", got: audio, want: try reference("audio_embeds"), counts: float32 == nil) && ok
            if let float32, let wantedVideo = float32["video_embeds_f32"], let wantedAudio = float32["audio_embeds_f32"] {
                let (video32, audio32) = try model.connect(states: referenceStates.map { $0.asType(.float32) },
                                                           mask: try reference("attention_mask"), float32: true)
                ok = Verify.report("connector video f32", got: video32, want: wantedVideo) && ok
                ok = Verify.report("connector audio f32", got: audio32, want: wantedAudio) && ok
            }
        }
        let wantedVideoText = try reference("video_embeds")
        let wantedAudioText = try reference("audio_embeds")

        // 2.5: the DurationHead on the reference's contexts, and its seconds-to-frames rule.
        if let wanted = references["duration_seconds"] {
            let got = try model.durationSeconds(video: wantedVideoText, audio: wantedAudioText)
            ok = Verify.report("duration seconds", got: got.asType(.float32), want: wanted) && ok
            print("    predicted \(got.asType(.float32).item(Float.self)) s")
        }
        if let table = references["duration_frames_table"] {
            let rows = table.asArray(Float.self)
            var mismatches = 0
            for row in stride(from: 0, to: rows.count, by: 3) {
                let fps = Double(rows[row])
                let got = LTXVideoModel.durationFrames(seconds: Double(rows[row + 1]), fps: fps,
                                                       minFrames: Int(fps.rounded(.toNearestOrEven)),
                                                       maxFrames: Int((10 * fps).rounded(.toNearestOrEven)))
                if got != Int(rows[row + 2]) { mismatches += 1 }
            }
            print("  \(mismatches == 0 ? "✓" : "✗") duration frames    \(rows.count / 3) durations\(mismatches == 0 ? "" : ", \(mismatches) DIFFERENT")")
            ok = mismatches == 0 && ok
        }

        // 4. Noise and positions.
        let videoInit = try reference("stage1_video_init")
        let audioInit = try reference("stage1_audio_init")
        // With a reference image, the first frame's tokens are the image's: the noise is compared after them.
        let noise = LTXVideoModel.stageOneNoise(shape: videoInit.shape, seed: seed)
        let skipped = references["stage1_image_tokens"]?.shape[1] ?? 0
        ok = Verify.report("stage 1 video noise", got: noise[0..., skipped...], want: videoInit[0..., skipped...]) && ok
        ok = Verify.report("stage 1 audio noise", got: LTXVideoModel.stageOneNoise(shape: audioInit.shape, seed: seed + 1), want: audioInit) && ok
        let upsampleIn = try reference("upsample_in")
        let (f, h1, w1) = (upsampleIn.shape[2], upsampleIn.shape[3], upsampleIn.shape[4])
        let positions1 = LTXVideoModel.positions(frames: f, height: h1, width: w1, fps: fps, audioTokens: audioInit.shape[1])
        ok = Verify.report("video positions", got: positions1.video, want: try reference("pass1_video_positions")) && ok
        ok = Verify.report("audio positions", got: positions1.audio, want: try reference("pass1_audio_positions")) && ok
        let positions2 = LTXVideoModel.positions(frames: f, height: h1 * 2, width: w1 * 2, fps: fps, audioTokens: audioInit.shape[1])
        // 2.5: the keyframe marker on the first latent frame of each stage.
        let keyframes = (stage1: h1 * w1, stage2: h1 * w1 * 4)
        for (stage, count) in [(1, keyframes.stage1), (2, keyframes.stage2)] {
            guard let wanted = references["pass\(stage)_video_keyframes_mask"] else { continue }
            let tokens = wanted.shape[1]
            let got = concatenated([MLXArray.ones([1, count, 1]), MLXArray.zeros([1, tokens - count, 1])], axis: 1)
            ok = Verify.report("keyframes mask \(stage)", got: got, want: wanted.asType(.float32)) && ok
        }
        // 2.5: the ancestral noise, video then audio at each of stage 1's steps but the last.
        var draws: [MLXArray] = []
        while let draw = references["ancestral_noise_\(draws.count)"] { draws.append(draw) }
        if !draws.isEmpty {
            let got = LTXVideoModel.ancestralNoise(shapes: draws.map(\.shape), seed: seed + LTXConfig.ancestralSeedOffset)
            for (index, draw) in draws.enumerated() where index < 2 || index == draws.count - 1 {
                ok = Verify.report("ancestral noise \(index)", got: got[index], want: draw) && ok
            }
        }

        // 5. The image encoder (image-to-video fixtures): in bfloat16 its convolutions carry
        // MLX's conv3d rounding, as the decoder's do (shown); in float32 the math decides.
        let float32Tokens = try? loadArrays(url: fixture.appending(path: "image_tokens_f32.safetensors"))
        for stage in [1, 2] where references["stage\(stage)_image_pixels"] != nil {
            let pixels = try reference("stage\(stage)_image_pixels")
            let tokens = try model.imageTokens(pixels: pixels)
            ok = Verify.report("image tokens \(stage)", got: tokens, want: try reference("stage\(stage)_image_tokens"),
                               counts: float32Tokens == nil) && ok
            if let wanted = float32Tokens?["stage\(stage)_image_tokens_f32"] {
                ok = Verify.report("image tokens \(stage) f32", got: try model.imageTokens(pixels: pixels, float32: true), want: wanted) && ok
            }
        }

        // 6. Stage 1: the first pass, then the loop, from the reference's states.
        let pass1 = try model.transformerPass(
            video: try reference("pass1_video_in"), audio: try reference("pass1_audio_in"), sigma: try reference("pass1_sigma"),
            videoTimesteps: references["pass1_video_timesteps"], videoText: wantedVideoText, audioText: wantedAudioText,
            videoPositions: positions1.video, audioPositions: positions1.audio, keyframeTokens: keyframes.stage1
        )
        ok = Verify.report("pass 1 video", got: pass1.video, want: try reference("pass1_video_velocity")) && ok
        ok = Verify.report("pass 1 audio", got: pass1.audio, want: try reference("pass1_audio_velocity")) && ok
        let stage1 = try model.denoiseStage(
            video: (videoInit, try reference("stage1_video_clean"), try reference("stage1_video_mask")),
            audio: (audioInit, try reference("stage1_audio_clean"), try reference("stage1_audio_mask")),
            sigmas: LTXConfig.distilledSigmas, videoText: wantedVideoText, audioText: wantedAudioText,
            videoPositions: positions1.video, audioPositions: positions1.audio, keyframeTokens: keyframes.stage1,
            ancestralSeed: ancestral ? seed + LTXConfig.ancestralSeedOffset : nil
        )
        // The first steps move the bfloat16 latent by about one unit of its last place, so which
        // way each rounds follows differences of 1e-5 in the velocity: two correct loops end a
        // percent or so apart. Each pass on the reference's own input (below) decides.
        ok = Verify.report("stage 1 video", got: stage1.video, want: try reference("stage1_video_out"), counts: false) && ok
        ok = Verify.report("stage 1 audio", got: stage1.audio, want: try reference("stage1_audio_out"), counts: false) && ok

        // Every pass of both loops on the reference's own inputs: whether a loop's drift comes
        // from one step or from rounding carried from step to step.
        let conditioned = references["stage1_image_tokens"] != nil
        let sigmaTable = Array(LTXConfig.distilledSigmas.dropLast()) + Array(LTXConfig.stage2Sigmas.dropLast())
        for step in 0 ..< 11 where references["step\(step)_video_in"] != nil {
            let positions = step < 8 ? positions1 : positions2
            // With a reference image the first frame's tokens run at timestep 0: `mask · σ`, the
            // mask float32 once conditioned, σ the table's own value.
            let mask = try reference(step < 8 ? "stage1_video_mask" : "stage2_video_mask")
            let timesteps: MLXArray? = conditioned ? (mask * Float(sigmaTable[step])).squeezed(axis: -1) : nil
            if let timesteps, step == 0 {
                ok = Verify.report("video timesteps", got: timesteps, want: try reference("pass1_video_timesteps")) && ok
            }
            let pass = try model.transformerPass(
                video: try reference("step\(step)_video_in"), audio: try reference("step\(step)_audio_in"),
                sigma: try reference("step\(step)_sigma"), videoTimesteps: timesteps, videoText: wantedVideoText, audioText: wantedAudioText,
                videoPositions: positions.video, audioPositions: positions.audio,
                keyframeTokens: step < 8 ? keyframes.stage1 : keyframes.stage2
            )
            ok = Verify.report("  step \(step) video", got: pass.video, want: try reference("step\(step)_video_velocity")) && ok
            ok = Verify.report("  step \(step) audio", got: pass.audio, want: try reference("step\(step)_audio_velocity")) && ok
        }

        // 7. The upsampler.
        let upsampled = try model.upsampleLatent(upsampleIn)
        ok = Verify.report("upsampler", got: upsampled, want: try reference("upsample_out")) && ok

        // 8. Stage 2's renoised states (text-to-video: no conditioning), its first pass and loop.
        if references["stage2_image_tokens"] == nil {
            let upTokens = try reference("upsample_out").transposed(0, 2, 3, 4, 1).reshaped([1, -1, 128])
            ok = Verify.report("stage 2 video noise", got: LTXVideoModel.stageTwoVideoNoise(upTokens, seed: seed),
                               want: try reference("stage2_video_init")) && ok
            ok = Verify.report("stage 2 audio noise", got: LTXVideoModel.stageTwoAudioNoise(try reference("stage1_audio_out"), seed: seed),
                               want: try reference("stage2_audio_init")) && ok
        }
        let pass2 = try model.transformerPass(
            video: try reference("pass2_video_in"), audio: try reference("pass2_audio_in"), sigma: try reference("pass2_sigma"),
            videoTimesteps: references["pass2_video_timesteps"], videoText: wantedVideoText, audioText: wantedAudioText,
            videoPositions: positions2.video, audioPositions: positions2.audio, keyframeTokens: keyframes.stage2
        )
        ok = Verify.report("pass 2 video", got: pass2.video, want: try reference("pass2_video_velocity")) && ok
        ok = Verify.report("pass 2 audio", got: pass2.audio, want: try reference("pass2_audio_velocity")) && ok
        let stage2 = try model.denoiseStage(
            video: (try reference("stage2_video_init"), try reference("stage2_video_clean"), try reference("stage2_video_mask")),
            audio: (try reference("stage2_audio_init"), try reference("stage2_audio_clean"), try reference("stage2_audio_mask")),
            sigmas: LTXConfig.stage2Sigmas, videoText: wantedVideoText, audioText: wantedAudioText,
            videoPositions: positions2.video, audioPositions: positions2.audio, keyframeTokens: keyframes.stage2
        )
        ok = Verify.report("stage 2 video", got: stage2.video, want: try reference("stage2_video_out")) && ok
        ok = Verify.report("stage 2 audio", got: stage2.audio, want: try reference("stage2_audio_out")) && ok

        // 9. The decoders.
        // In bfloat16 the decoder's twenty-odd convolutions carry the rounding of MLX's conv3d,
        // which changed between the two versions (shown); in float32 the math decides.
        let pixels = try model.decodePixels(try reference("video_latent"))
        ok = Verify.report("video decode", got: pixels, want: try reference("pixels"), counts: false) && ok
        let float32URL = fixture.appending(path: "pixels_f32.safetensors")
        if let wanted = try? loadArrays(url: float32URL)["pixels_f32"] {
            let pixels32 = try model.decodePixels(try reference("video_latent"), float32: true)
            ok = Verify.report("video decode f32", got: pixels32, want: wanted) && ok
        }
        let stagesURL = fixture.appending(path: "decode_stages.safetensors")
        if FileManager.default.fileExists(atPath: stagesURL.path) {
            let wanted = try loadArrays(url: stagesURL)
            for (name, value) in try model.decoderStages(try reference("video_latent")) {
                if let want = wanted[name] { _ = Verify.report("  \(name)", got: value, want: want, counts: false) }
            }
            let frames = pixels.shape[2]
            for frame in 0 ..< frames {
                _ = Verify.report("  frame \(frame)", got: pixels[0..., 0..., frame], want: try reference("pixels")[0..., 0..., frame], counts: false)
            }
        }
        let mel = try model.decodeMel(try reference("audio_latent"))
        ok = Verify.report("audio decode", got: mel, want: try reference("mel")) && ok
        let waveform = try model.vocode(try reference("mel"))
        ok = Verify.report("vocoder", got: waveform, want: try reference("waveform")) && ok
        return ok
    }
}
