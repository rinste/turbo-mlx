import Foundation
import MLX
import TurboEngineCore

/// The MOSS-SoundEffect port (Engine/Fixtures/make_moss_soundeffect_fixture.py) on the real weights. The prompt: the
/// pipeline's text, the cleaned text and the token ids of a corpus, exactly, and the samples a duration keeps. The
/// schedule against torch's. The encoder in float32 against transformers' Qwen3 in float32, then in bf16 (shown).
/// With OpenMOSS's float32 DiT, against the goldens captured from OpenMOSS's code: the velocity at t = 1000 and 500
/// from the golden noise and prompt states, the ten guided steps, and the pack's VAE decoding the golden latents.
/// Then the pack's bf16 DiT on the same inputs, judged against the same float32 goldens.
enum VerifyMossSoundEffect {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        var ok = true
        func check(_ stage: String, _ got: MLXArray, _ want: MLXArray, counts: Bool = true) -> Bool {
            eval(got)
            let pass = Verify.report(stage, got: got, want: want, counts: counts)
            print(String(format: "      cos %.6f", VerifyMusic3.cosine(got, want)))
            return pass
        }
        /// A bf16 stage: shown against the float32 truth, failing only past `limit` relative RMS.
        func judged(_ stage: String, _ got: MLXArray, _ truth: MLXArray, limit: Float) -> Bool {
            _ = check(stage, got, truth, counts: false)
            let distance = VerifyMusic3.relativeRMS(got, truth)
            let pass = distance <= limit
            print(String(format: "      from float32: %.5f (limit %.3f)%@", distance, limit, pass ? "" : "  ✗ FURTHER"))
            return pass
        }

        let pack = URL(fileURLWithPath: json["pack"] as? String ?? fixture.path)
        print("loading \(pack.path) with OpenMOSS's float32 DiT…")
        let official = (json["official_transformer"] as? String).map { URL(fileURLWithPath: $0) }
        var model: MossSoundEffectModel? = try MossSoundEffectModel(folder: pack, transformerFile: official)

        // 1. The prompt.
        let corpus = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.appending(path: "prompts.json"))) as? [String: Any] ?? [:]
        var promptsOK = true
        for item in corpus["prompts"] as? [[String: Any]] ?? [] {
            let prompt = item["prompt"] as? String ?? ""
            let text = (item["seconds"] as? NSNumber).map { MossSoundEffectPrompt.withDuration(prompt, seconds: $0.doubleValue) } ?? prompt
            let sameText = Array(text.unicodeScalars) == Array((item["text"] as? String ?? "").unicodeScalars)
            let cleaned = MossSoundEffectPrompt.clean(text)
            let sameClean = Array(cleaned.unicodeScalars) == Array((item["clean"] as? String ?? "").unicodeScalars)
            let ids = try model!.tokenIds(text)
            let sameIds = ids == (item["ids"] as? [Int] ?? [])
            if !(sameText && sameClean && sameIds) {
                promptsOK = false
                print("  ✗ \(prompt.debugDescription.prefix(60)): text \(sameText), clean \(sameClean) (\(cleaned.debugDescription.prefix(80))), ids \(sameIds)")
            }
        }
        for item in corpus["durations"] as? [[String: Any]] ?? [] {
            let seconds = (item["seconds"] as? NSNumber)?.doubleValue ?? 0
            let text = MossSoundEffectPrompt.withDuration("x", seconds: seconds)
            let samples = Int(48_000 * MossSoundEffectModel.rounded(seconds))
            if text != item["text"] as? String || samples != item["samples"] as? Int {
                promptsOK = false
                print("  ✗ \(seconds) s: \(text), \(samples) samples; expected \(item["text"] ?? ""), \(item["samples"] ?? "")")
            }
        }
        print("  prompt texts, cleaning, tokens and durations: \(promptsOK ? "identical" : "DIFFERENT")")
        ok = promptsOK && ok

        // 2. The schedule.
        for steps in [1, 2, 10, 50, 100, 200] {
            let sigmas = MLXArray(MossSoundEffectModel.sigmas(steps: steps, shift: model!.config.shift))
            ok = check("sigmas \(steps)", sigmas, try reference("sigmas\(steps)")) && ok
        }

        // 3. The encoder, in float32 then bf16, from the fixture's ids.
        for index in 0 ..< ((json["encoder_prompts"] as? [String])?.count ?? 0) {
            let mask = try reference("encoder\(index)_mask")
            let valid = mask.sum().item(Int.self)
            let ids = try reference("encoder\(index)_ids")[0 ..< valid].asArray(Int32.self).map(Int.init)
            let text = (json["encoder_prompts"] as? [String])?[index] ?? ""
            let sameIds = try model!.tokenIds(text) == ids
            print("  encoder prompt \(index) ids: \(sameIds ? "identical" : "DIFFERENT")")
            ok = sameIds && ok
            let truth = try reference("encoder\(index)_states")
            ok = check("encoder \(index) f32", try model!.encoderStates(ids: ids, computeType: .float32), truth) && ok
            // bf16 rounding through 28 layers: the port passes when it is no further from float32 than transformers'
            // bf16 run is, give or take half (the two round in different places).
            let theirs = VerifyMusic3.relativeRMS(try reference("encoder\(index)_states_bf16"), truth)
            print(String(format: "      (transformers in bf16: %.5f from float32)", theirs))
            ok = judged("encoder \(index) bf16", try model!.encoderStates(ids: ids), truth, limit: 1.5 * theirs + 1e-4) && ok
        }

        // 4. The DiT and the loop in float32, from the goldens' noise and states.
        let noise = try reference("golden_noise")
        let golden = json["golden"] as? [String: Any] ?? [:]
        let steps = (golden["steps"] as? NSNumber)?.intValue ?? 10
        let guidance = (golden["cfg"] as? NSNumber)?.floatValue ?? 4
        func dit(_ model: MossSoundEffectModel, precision: String) throws -> (MLXArray, MLXArray, MLXArray) {
            let prompt = model.transformer.embedText(try reference("golden_context"))
            let negative = model.transformer.embedText(try reference("golden_context_nega"))
            let v1000 = model.transformer(noise, timestep: 1000, text: prompt).asType(.float32)
            let v500 = model.transformer(noise, timestep: 500, text: prompt).asType(.float32)
            eval(v1000, v500)
            let started = Date()
            let latents = try model.denoise(noise: noise, prompt: prompt, negative: negative, steps: steps, guidance: guidance)
            print(String(format: "  (%@: %d guided steps in %.1f s)", precision, steps, Date().timeIntervalSince(started)))
            return (v1000, v500, latents)
        }
        let (v1000, v500, latents) = try dit(model!, precision: "float32")
        ok = check("velocity t=1000", v1000, try reference("golden_velocity_t1000")) && ok
        ok = check("velocity t=500", v500, try reference("golden_velocity_t500")) && ok
        ok = check("10 guided steps", latents, try reference("golden_final_latent")) && ok

        // 5. The VAE (float32 in the pack) on the golden latents.
        let wantAudio = try reference("golden_audio").reshaped([1, -1])
        let decodeStarted = Date()
        let audio = model!.vae.decode(try reference("golden_final_latent"))
        eval(audio)
        print(String(format: "  (decode of %d latents in %.1f s)", latents.dim(2), Date().timeIntervalSince(decodeStarted)))
        ok = check("decode", audio, wantAudio) && ok
        // The pipeline decodes only the latents a shorter clip's samples depend on.
        for seconds in [10.0, 2.5, 0.1] {
            let kept = Int(48_000 * seconds)
            let cropped = model!.decode(try reference("golden_final_latent"), seconds: seconds).reshaped([1, -1])
            ok = check("decode \(seconds) s", cropped, wantAudio[0..., 0 ..< kept]) && ok
        }
        model = nil
        Memory.clearCache()

        // 6. The pack's bf16 DiT, judged against the float32 goldens.
        print("loading \(pack.path) (bf16 DiT)…")
        let packModel = try MossSoundEffectModel(folder: pack)
        let (b1000, b500, bLatents) = try dit(packModel, precision: "bf16")
        ok = judged("bf16 velocity 1000", b1000, try reference("golden_velocity_t1000"), limit: 0.03) && ok
        ok = judged("bf16 velocity 500", b500, try reference("golden_velocity_t500"), limit: 0.03) && ok
        ok = judged("bf16 10 steps", bLatents, try reference("golden_final_latent"), limit: 0.05) && ok
        let bAudio = packModel.vae.decode(bLatents)
        ok = judged("bf16 decode", bAudio, wantAudio, limit: 0.1) && ok
        return ok
    }
}
