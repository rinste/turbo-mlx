import Foundation
import MLX
import TurboEngineCore

/// Compares the port with the outputs mflux computed on the same small checkpoint
/// (Engine/Fixtures/make_klein_fixture.py): the text encoder, one transformer pass, the whole
/// denoising loop with the scheduler, and the VAE decode. Prints the largest error of each stage
/// relative to the reference's scale and fails above the tolerance.
enum Verify {
    static let tolerance: Float = 0.03

    static func run(fixture: URL) -> Bool {
        do {
            let data = try Data(contentsOf: fixture.appending(path: "fixture.json"))
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                print("fixture.json is not an object"); return false
            }
            let config = KleinConfig.fixture(json)
            let references = try loadArrays(url: fixture.appending(path: "references.safetensors"))
            func reference(_ name: String) throws -> MLXArray {
                guard let array = references[name] else { throw VerifyError.missingReference(name) }
                return array
            }

            print("loading \(fixture.path) as \(config.name)…")
            let model = try KleinModel(modelPath: fixture, config: config, loadTokenizer: false)
            var ok = true

            // 1. Text encoder.
            let encoded = model.encode(inputIds: try reference("input_ids"), attentionMask: try reference("attention_mask"))
            ok = report("text encoder", got: encoded.embeds, want: try reference("prompt_embeds")) && ok
            ok = report("text ids", got: encoded.ids, want: try reference("text_ids")) && ok

            // 2. Initial noise and ids for the fixture's seed and size.
            let width = (json["width"] as? NSNumber)?.intValue ?? 128
            let height = (json["height"] as? NSNumber)?.intValue ?? 128
            let seed = (json["seed"] as? NSNumber)?.intValue ?? 0
            let steps = (json["steps"] as? NSNumber)?.intValue ?? 4
            let initial = KleinModel.initialLatents(width: width, height: height, seed: seed)
            ok = report("initial noise", got: initial.latents, want: try reference("latents")) && ok
            ok = report("latent ids", got: initial.ids, want: try reference("latent_ids")) && ok

            // 3. One transformer pass at the reference's timestep (a value in [0, 1], scaled inside).
            let timestep = try reference("timestep").asType(.float32).item(Float.self)
            let noise = model.transformer(latents: try reference("latents"), prompt: try reference("prompt_embeds"), timestep: timestep, imageIds: try reference("latent_ids"), textIds: try reference("text_ids"))
            eval(noise)
            ok = report("transformer pass", got: noise, want: try reference("noise")) && ok

            // 4. The scheduler and the loop.
            let schedule = FlowMatchSchedule(steps: steps)
            ok = report("sigmas", got: MLXArray(schedule.sigmas), want: try reference("sigmas")) && ok
            ok = report("timesteps", got: MLXArray(schedule.timesteps), want: try reference("timesteps")) && ok
            var latents = try reference("latents")
            let prompt = try reference("prompt_embeds")
            for t in 0 ..< steps {
                let predicted = model.transformer(latents: latents, prompt: prompt, timestep: schedule.timesteps[t], imageIds: initial.ids, textIds: encoded.ids)
                latents = KleinModel.step(latents: latents, noise: predicted, schedule: schedule, index: t)
                eval(latents)
            }
            ok = report("denoised latents", got: latents, want: try reference("final_latents")) && ok

            // 5. VAE decode (the reference is channels first).
            let grid = try reference("final_latents").reshaped([1, initial.latentHeight, initial.latentWidth, 128])
            let decoded = model.vae.decodePacked(grid)
            eval(decoded)
            let wanted = try reference("decoded").transposed(0, 2, 3, 1)
            ok = report("vae decode", got: decoded, want: wanted) && ok

            print(ok ? "OK: the port matches mflux on this checkpoint" : "FAILED: see the stages above")
            return ok
        } catch {
            print("verify failed: \(error)")
            return false
        }
    }

    /// Largest absolute difference over the largest reference magnitude.
    static func report(_ stage: String, got: MLXArray, want: MLXArray) -> Bool {
        guard got.shape == want.shape else {
            print("  \(stage): shape \(got.shape), expected \(want.shape)")
            return false
        }
        let a = got.asType(.float32)
        let b = want.asType(.float32)
        let difference = abs(a - b).max().item(Float.self)
        let scale = Swift.max(abs(b).max().item(Float.self), 1e-6)
        let relative = difference / scale
        let pass = relative <= tolerance
        print(String(format: "  %@ %-18@ max |Δ| %.5f  (%.3f of the reference's %.4f)", pass ? "✓" : "✗", stage, difference, relative, scale))
        return pass
    }

    enum VerifyError: LocalizedError {
        case missingReference(String)
        var errorDescription: String? {
            switch self { case .missingReference(let name): "references.safetensors has no \(name)" }
        }
    }
}
