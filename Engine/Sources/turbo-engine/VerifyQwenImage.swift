import Foundation
import MLX
import TurboEngineCore

/// The Qwen-Image port against mflux's outputs on a small checkpoint
/// (Engine/Fixtures/make_qwen_image_fixture.py): the text encoder on both prompts, the initial
/// noise, one transformer pass, the schedule, the guided loop and the VAE decode.
enum VerifyQwenImage {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let config = QwenImageConfig.fixture(json)
        print("loading \(fixture.path) as \(config.name)…")
        let model = try QwenImageModel(modelPath: fixture, config: config, loadTokenizer: false)
        var ok = true

        // 1. Text encoder: the final-norm states after the template's tokens, for both prompts.
        func ids(_ name: String) throws -> [Int] { try reference(name).asArray(Int32.self).map { Int($0) } }
        let prompt = try model.promptEmbeds(ids: try ids("input_ids"))
        ok = Verify.report("text encoder", got: prompt, want: try reference("prompt_embeds")) && ok
        let negative = try model.promptEmbeds(ids: try ids("negative_input_ids"))
        ok = Verify.report("negative prompt", got: negative, want: try reference("negative_prompt_embeds")) && ok

        // 2. Initial noise for the fixture's seed and size.
        let width = (json["width"] as? NSNumber)?.intValue ?? 64
        let height = (json["height"] as? NSNumber)?.intValue ?? 64
        let seed = (json["seed"] as? NSNumber)?.intValue ?? 0
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 3
        let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 4
        let (latentHeight, latentWidth) = (height / 16, width / 16)
        let initial = QwenImageModel.initialLatents(width: width, height: height, seed: seed)
        ok = Verify.report("initial noise", got: initial, want: try reference("latents")) && ok

        // 3. One transformer pass at the reference's timestep.
        let (transformer, vae) = try model.loadedImageSide()
        let timestep = try reference("timestep").asType(.float32).item(Float.self)
        let embeds = try reference("prompt_embeds")
        let noise = transformer(latents: try reference("latents"), prompt: embeds, timestep: timestep, latentHeight: latentHeight, latentWidth: latentWidth)
        eval(noise)
        ok = Verify.report("transformer pass", got: noise, want: try reference("noise")) && ok
        ok = try VerifyLoRA.run(fixture: fixture, model: model, plain: noise) {
            transformer(latents: try reference("latents"), prompt: embeds, timestep: timestep, latentHeight: latentHeight, latentWidth: latentWidth)
        } && ok

        // 4. The schedule and the guided loop.
        let schedule = LinearSchedule(steps: steps, width: width, height: height, shift: config.shift)
        ok = Verify.report("sigmas", got: MLXArray(schedule.sigmas), want: try reference("sigmas")) && ok
        var latents = try reference("latents")
        let negativeEmbeds = try reference("negative_prompt_embeds")
        for t in 0 ..< steps {
            let positive = transformer(latents: latents, prompt: embeds, timestep: schedule.sigmas[t], latentHeight: latentHeight, latentWidth: latentWidth)
            let unconditional = transformer(latents: latents, prompt: negativeEmbeds, timestep: schedule.sigmas[t], latentHeight: latentHeight, latentWidth: latentWidth)
            let guided = QwenImageModel.guidedNoise(positive, negative: unconditional, guidance: guidance)
            latents = schedule.step(latents: latents, noise: guided, index: t)
            eval(latents)
        }
        ok = Verify.report("denoised latents", got: latents, want: try reference("final_latents")) && ok

        // 4b. The guided loop with the latents, and so the stream, started in bf16 (the app's
        // "16-bit precision" option): shown, not decided, since the option moves the pixels by design.
        var half = try reference("latents").asType(.bfloat16)
        for t in 0 ..< steps {
            let positive = transformer(latents: half, prompt: embeds, timestep: schedule.sigmas[t], latentHeight: latentHeight, latentWidth: latentWidth)
            let unconditional = transformer(latents: half, prompt: negativeEmbeds, timestep: schedule.sigmas[t], latentHeight: latentHeight, latentWidth: latentWidth)
            half = schedule.step(latents: half, noise: QwenImageModel.guidedNoise(positive, negative: unconditional, guidance: guidance), index: t)
            eval(half)
        }
        _ = Verify.report("16-bit stream loop", got: half, want: try reference("final_latents"), counts: false)

        // 5. VAE decode (the reference is channels first).
        let grid = QwenImageModel.unpack(try reference("final_latents"), latentHeight: latentHeight, latentWidth: latentWidth)
        let decoded = vae.decode(grid)
        eval(decoded)
        ok = Verify.report("vae decode", got: decoded, want: try reference("decoded").transposed(0, 2, 3, 1)) && ok
        return ok
    }
}
