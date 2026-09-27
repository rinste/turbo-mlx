import Foundation
import MLX
import TurboEngineCore

/// The Ming-Image port against mflux's outputs on a small checkpoint
/// (Engine/Fixtures/make_ming_fixture.py): the whole text side (encoder, connector, heads), the
/// initial noise, one transformer pass with and without the condition, the schedule, the guided
/// loop and the RGBA decode.
enum VerifyMing {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let config = MingConfig.fixture(json)
        print("loading \(fixture.path) as \(config.name)…")
        let model = try MingModel(modelPath: fixture, config: config, loadTokenizer: false)
        var ok = true

        // 1. The text side: caption features from the query tokens, direct-VLM tokens from the prompt.
        // The router picks experts from bf16 scores, so a rounding difference between MLX versions
        // can change a choice; the fixture's random weights make that unlikely but not impossible.
        let promptIds = try reference("prompt_ids").asArray(Int32.self).map { Int($0) }
        let (capFeats, capFeats2) = try model.encode(promptIds: promptIds)
        ok = Verify.report("caption features", got: capFeats, want: try reference("cap_feats")) && ok
        ok = Verify.report("direct-VLM tokens", got: capFeats2, want: try reference("cap_feats_2")) && ok

        // 2. Initial noise for the fixture's seed and size.
        let width = (json["width"] as? NSNumber)?.intValue ?? 64
        let height = (json["height"] as? NSNumber)?.intValue ?? 64
        let seed = (json["seed"] as? NSNumber)?.intValue ?? 0
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 4
        let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 1
        let initial = MingModel.initialLatents(width: width, height: height, seed: seed)
        ok = Verify.report("initial noise", got: initial, want: try reference("latents")) && ok

        // 3. One transformer pass at the reference's timestep, conditional and unconditional.
        let (transformer, vae) = try model.loadedImageSide()
        let latents = try reference("latents")
        let timestep = try reference("timestep")
        let refCap = try reference("cap_feats")
        let refCap2 = try reference("cap_feats_2")
        let x = latents[0].expandedDimensions(axis: 1)
        let noise = transformer(latents: x, timestep: timestep, capFeats: refCap, extraCaption: refCap2)
        eval(noise)
        ok = Verify.report("transformer pass", got: noise, want: try reference("noise")) && ok
        let unconditional = transformer(latents: x, timestep: timestep, capFeats: MLXArray.zeros(like: refCap), extraCaption: MLXArray.zeros(like: refCap2))
        eval(unconditional)
        ok = Verify.report("unconditional pass", got: unconditional, want: try reference("noise_unconditional")) && ok

        // 4. The static-shift schedule and the guided loop.
        let sigmas = StaticShiftSchedule(steps: steps, shift: config.sigmaShift).sigmas
        ok = Verify.report("sigmas", got: MLXArray(sigmas), want: try reference("sigmas")) && ok
        var current = latents
        for t in 0 ..< steps {
            let (sigma, next) = (sigmas[t], sigmas[t + 1])
            if sigma > 0 {
                let velocity = MingModel.predict(transformer: transformer, latents: current, sigma: sigma, capFeats: refCap, capFeats2: refCap2, guidance: guidance)
                current = current + MLXArray(next - sigma) * velocity.asType(.float32)
            }
            eval(current)
        }
        ok = Verify.report("denoised latents", got: current, want: try reference("final_latents")) && ok

        // 5. RGBA decode (the reference is channels first).
        let grid = try reference("final_latents").asType(.bfloat16).transposed(0, 2, 3, 1)
        let decoded = vae.decode(grid)
        eval(decoded)
        ok = Verify.report("vae decode", got: decoded, want: try reference("decoded").transposed(0, 2, 3, 1)) && ok
        return ok
    }
}
