import Foundation
import MLX
import TurboEngineCore

/// The MiniMax Music 3 port against mlx-audio on the pack itself (Engine/Reference/music3_reference.py --record), each
/// stage from the reference's own inputs. The prompt: the text assembled and its ids, exactly (and the official prompt
/// functions on a corpus, from prompts.json when the folder has one). The autoregressive stage: the prompt's last hidden
/// states, then for the first frames, teacher-forced on the reference's codes, the guided semantic logits, the token
/// drawn with the reference's key, the residual codes (exact), the depth decoder's hidden states, the feedback and the
/// next hidden states; then every frame from the seed, and how many of the frames' codes agree. The flow: the first
/// window's condition, the noise from seed + 7 (exact), both velocities at one time, the window's latents after every
/// step, the vocoder's waveform, and the two-window run with its overlap.
enum VerifyMusic3 {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        var ok = true
        func check(_ stage: String, _ got: MLXArray, _ want: MLXArray, counts: Bool = true) -> Bool {
            eval(got)
            let pass = Verify.report(stage, got: got, want: want, counts: counts)
            print(String(format: "      cos %.6f", cosine(got, want)))
            return pass
        }
        // With float32 references (`--truth`), a stage passes when the port is no further from them than the
        // reference is, give or take half: the PyPI and mlx-swift builds of MLX round bf16 apart.
        let truths = (try? loadArrays(url: fixture.appending(path: "truth.safetensors"))) ?? [:]
        func judged(_ stage: String, _ got: MLXArray, _ want: MLXArray, truth name: String) -> Bool {
            guard let truth = truths[name] else { return check(stage, got, want) }
            _ = check(stage, got, want, counts: false)
            let ours = relativeRMS(got, truth), theirs = relativeRMS(want, truth)
            let pass = ours <= 1.5 * theirs + 1e-4
            print(String(format: "      from float32: ours %.5f, mlx-audio's %.5f%@", ours, theirs, pass ? "" : "  ✗ FURTHER"))
            return pass
        }
        func exact(_ stage: String, _ got: MLXArray, _ want: MLXArray) -> Bool {
            let same = got.shape == want.shape && (got.asType(.int64) .== want.asType(.int64)).all().item(Bool.self)
            print("  \(stage): \(same ? "identical" : "DIFFERENT \(got.shape) vs \(want.shape)")")
            return same
        }

        if FileManager.default.fileExists(atPath: fixture.appending(path: "prompts.json").path) {
            ok = try verifyPrompts(fixture.appending(path: "prompts.json")) && ok
        }

        let pack = URL(fileURLWithPath: json["pack"] as? String ?? fixture.path)
        print("loading \(pack.path)…")
        let model = try Music3Model(folder: pack)
        let config = model.config
        let seed = (json["seed"] as? NSNumber)?.intValue ?? 7

        // 1. The prompt.
        let caption = json["caption"] as? String ?? "", lyrics = json["lyrics"] as? String ?? ""
        let assembled = Music3Prompt.assemble(caption: caption, lyrics: lyrics)
        let sameText = Array(assembled.unicodeScalars) == Array((json["assembled"] as? String ?? "").unicodeScalars)
        print("  prompt text: \(sameText ? "identical" : "DIFFERENT")")
        ok = sameText && ok
        let ids = try model.promptIDs(caption: caption, lyrics: lyrics)
        ok = exact("prompt ids", ids, try reference("ids")) && ok

        // 2. The autoregressive stage, teacher-forced.
        let ar = try model.languageModels()
        let refIDs = try reference("ids")
        var (last, caches) = ar.prefill(ids: refIDs)
        ok = judged("prompt's last hidden states", last, try reference("prefill_last"), truth: "prefill_last") && ok
        let arSteps = (json["ar_steps"] as? NSNumber)?.intValue ?? 0
        var codesAgreeing = 0, codesDrawn = 0
        for index in 0 ..< arSteps {
            let given = try reference("ar\(index)_last")
            if index > 0 { ok = check("frame \(index): hidden states in", last, given) && ok }
            let guided = ar.semanticLogits(given)
            let refGuided = try reference("ar\(index)_guided")
            let window = config.audioCodeOffset ..< config.audioCodeOffset + config.semanticVocabSize
            ok = check("frame \(index): guided semantic logits", guided[0..., window], refGuided[0..., window]) && ok
            let (sampled, next) = Music3AR.sampleTopK(refGuided, key: try reference("ar\(index)_key"))
            let refCodes = try reference("ar\(index)_codes")
            ok = exact("frame \(index): semantic token", sampled - Int32(config.audioCodeOffset), refCodes[0 ..< 1, 0]) && ok
            // The residual codes from the reference's key: a near tie may draw another code, so they are counted, not
            // judged; the depth decoder is judged on the reference's codes.
            let (codes, _, _) = ar.depthCodes(lastHidden: given, semanticCode: refCodes[0..., 0], key: next)
            let agreeing = (codes[0, 1...] .== refCodes[0, 1...].asType(codes.dtype)).asType(.int32).sum().item(Int.self)
            codesAgreeing += agreeing
            codesDrawn += config.residualCodebooks
            print("  frame \(index): residual codes drawn as the reference's: \(agreeing) of \(config.residualCodebooks)")
            ok = judged("frame \(index): depth hidden states", ar.depthHidden(lastHidden: given, codes: refCodes),
                        try reference("ar\(index)_depth_hidden"), truth: "ar\(index)_depth_hidden") && ok
            let feedback = ar.feedback(refCodes)
            ok = check("frame \(index): feedback", feedback, try reference("ar\(index)_feedback")) && ok
            last = ar.step(try reference("ar\(index)_feedback"), caches: caches)
        }
        if arSteps > 0 {
            ok = check("frame \(arSteps): hidden states in", last, try reference("ar_next_last")) && ok
            print("  residual codes drawn as the reference's: \(codesAgreeing) of \(codesDrawn)")
        }

        // 3. Every frame from the seed: identical while the draws agree.
        let refFrames = try reference("frames")
        let started = Date()
        let frames = try model.frames(ids: refIDs, maxFrames: (json["frames"] as? NSNumber)?.intValue ?? refFrames.dim(1), seed: seed)
        print(String(format: "  (%d frames in %.1f s)", frames.dim(1), Date().timeIntervalSince(started)))
        if frames.shape == refFrames.shape {
            let perFrame = abs(frames.asType(.float32) - refFrames.asType(.float32)).max(axes: [0, 2])
            let agree = (perFrame .== 0).asType(.int32).sum().item(Int.self)
            print("  frames identical to the reference's: \(agree) of \(frames.dim(1))")
        }
        _ = check("frames from the seed", frames, refFrames, counts: false)

        // 4. The flow, on the reference's frames.
        let condition = model.conditionEncoder(refFrames[0..., 0 ..< min(200, refFrames.dim(1))])
        ok = judged("condition", condition, try reference("condition"), truth: "condition") && ok
        let refCondition = try reference("condition")
        let noise = MLXRandom.normal([1, config.ditInChannels, refCondition.dim(1)], key: MLXRandom.RandomState(seed: UInt64(seed + 7)))
            .asType(refCondition.dtype)
        ok = exact("noise from seed + 7", noise.asType(.float32).view(dtype: .int32), try reference("noise").asType(.float32).view(dtype: .int32)) && ok
        let refNoise = try reference("noise")
        let t = (json["timestep"] as? NSNumber)?.floatValue ?? 0.4
        let timestep = MLXArray([t]).asType(refNoise.dtype)
        ok = judged("velocity, conditional", model.transformer(refNoise, timestep: timestep, condition: refCondition),
                    try reference("velocity_cond"), truth: "velocity_cond") && ok
        ok = judged("velocity, unconditional", model.transformer(refNoise, timestep: timestep, condition: zeros(like: refCondition)),
                    try reference("velocity_uncond"), truth: "velocity_uncond") && ok
        let flow = Music3Flow(conditionEncoder: model.conditionEncoder, transformer: model.transformer, config: config)
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 30
        let (latents, _) = try flow.denoise(noise: refNoise, condition: refCondition, steps: steps, previous: nil)
        ok = judged("latents after \(steps) steps", latents, try reference("latents"), truth: "latents") && ok
        ok = judged("vocoder", model.vocoder(try reference("latents")), try reference("wave"), truth: "wave") && ok

        // 5. Two windows and their overlap.
        let longSteps = (json["long_steps"] as? NSNumber)?.intValue ?? 4
        let chunks = try model.latents(frames: try reference("long_frames"), steps: longSteps, seed: seed)
        ok = judged("two windows, \(longSteps) steps each", try model.decode(chunks), try reference("long_audio"), truth: "long_audio") && ok
        return ok
    }

    /// The official prompt functions on a corpus: the cleaned caption, the normalized lyrics and the assembled text,
    /// code point for code point.
    static func verifyPrompts(_ url: URL) throws -> Bool {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any], let cases = json["cases"] as? [[String: String]]
        else { return false }
        var failing: Set<Int> = []
        for (index, item) in cases.enumerated() {
            let caption = item["caption"] ?? "", lyrics = item["lyrics"] ?? ""
            for (name, got, want) in [
                ("caption", Music3Prompt.cleanCaption(caption), item["caption_clean"] ?? ""),
                ("lyrics", Music3Prompt.normalizeLyrics(lyrics), item["lyrics_normalized"] ?? ""),
                ("assembled", Music3Prompt.assemble(caption: caption, lyrics: lyrics), item["assembled"] ?? ""),
            ] where Array(got.unicodeScalars) != Array(want.unicodeScalars) {
                failing.insert(index)
                print("  prompt case \(index) \(name): got \(got.debugDescription), want \(want.debugDescription)")
            }
        }
        print("  prompt corpus: \(cases.count - failing.count) of \(cases.count) cases as the official functions make them")
        return failing.isEmpty
    }

    /// RMS of the difference over the RMS of `b`.
    static func relativeRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return .nan }
        let x = a.asType(.float32), y = b.asType(.float32)
        return (sqrt(square(x - y).mean()) / (sqrt(square(y).mean()) + 1e-12)).item(Float.self)
    }

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return .nan }
        let x = a.asType(.float32).reshaped(-1), y = b.asType(.float32).reshaped(-1)
        return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()) + 1e-12)).item(Float.self)
    }
}
