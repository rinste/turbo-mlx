import Foundation
import MLX
import TurboEngineCore

/// The Z-Image port against mflux's outputs on a small checkpoint
/// (Engine/Fixtures/make_zimage_fixture.py): the text encoder, the initial noise, one transformer
/// pass, the schedule, the whole loop and the VAE decode.
enum VerifyZImage {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let config = ZImageConfig.fixture(json)
        print("loading \(fixture.path) as \(config.name)…")
        let model = try ZImageModel(modelPath: fixture, config: config, loadTokenizer: false)
        var ok = true

        // 1. Text encoder: the second-to-last hidden state of the real tokens, computed in float32.
        let encoded = model.encode(inputIds: try reference("input_ids"))
        ok = Verify.report("text encoder", got: encoded, want: try reference("prompt_embeds")) && ok

        // 2. Initial noise for the fixture's seed and size.
        let width = (json["width"] as? NSNumber)?.intValue ?? 128
        let height = (json["height"] as? NSNumber)?.intValue ?? 128
        let seed = (json["seed"] as? NSNumber)?.intValue ?? 0
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 4
        let initial = ZImageModel.initialLatents(width: width, height: height, seed: seed)
        ok = Verify.report("initial noise", got: initial, want: try reference("latents")) && ok

        // 3. One transformer pass at the reference's timestep (1 − σ).
        let prompt = try reference("prompt_embeds")
        let noise = model.transformer(latents: try reference("latents"), timestep: try reference("timestep"), capFeats: prompt)
        eval(noise)
        ok = Verify.report("transformer pass", got: noise, want: try reference("noise")) && ok
        ok = try VerifyLoRA.run(fixture: fixture, model: model, plain: noise) {
            model.transformer(latents: try reference("latents"), timestep: try reference("timestep"), capFeats: prompt)
        } && ok

        // 4. The linear schedule, shifted for the size, and the loop.
        let schedule = LinearSchedule(steps: steps, width: width, height: height, shift: config.shift)
        ok = Verify.report("sigmas", got: MLXArray(schedule.sigmas), want: try reference("sigmas")) && ok
        var latents = try reference("latents")
        for t in 0 ..< steps {
            let predicted = model.transformer(latents: latents, timestep: MLXArray([1 - schedule.sigmas[t]]), capFeats: prompt)
            latents = schedule.step(latents: latents, noise: predicted, index: t)
            eval(latents)
        }
        ok = Verify.report("denoised latents", got: latents, want: try reference("final_latents")) && ok

        // 4b. The same loop with the stream kept in the weights' precision (the app's "16-bit
        // precision" option): shown, not decided, since the option moves the pixels by design.
        model.transformer.setKeepsWeightsPrecision(true)
        var half = try reference("latents")
        for t in 0 ..< steps {
            let predicted = model.transformer(latents: half, timestep: MLXArray([1 - schedule.sigmas[t]]), capFeats: prompt)
            half = schedule.step(latents: half, noise: predicted, index: t)
            eval(half)
        }
        model.transformer.setKeepsWeightsPrecision(false)
        _ = Verify.report("16-bit stream loop", got: half, want: try reference("final_latents"), counts: false)

        // 5. VAE decode (the reference is channels first).
        let final = try reference("final_latents")
        let grid = final.reshaped([1, final.shape[0], final.shape[2], final.shape[3]]).transposed(0, 2, 3, 1)
        let decoded = model.vae.decode(grid)
        eval(decoded)
        ok = Verify.report("vae decode", got: decoded, want: try reference("decoded").transposed(0, 2, 3, 1)) && ok
        return ok
    }
}
