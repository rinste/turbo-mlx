import Foundation
import MLX
import TurboEngineCore

/// Compares a port with the outputs mflux computed on the same small checkpoint (see
/// Engine/Fixtures/make_*_fixture.py): stage by stage, the largest error relative to the
/// reference's scale, failing above the tolerance. `fixture.json` names the family.
enum Verify {
    static let tolerance: Float = 0.03

    /// Every fixture in `folder` (the sub-folders with a `fixture.json`), one after the other.
    static func runAll(folder: URL) -> Bool {
        let entries = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        let fixtures = entries.filter { FileManager.default.fileExists(atPath: $0.appending(path: "fixture.json").path) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !fixtures.isEmpty else {
            print("no fixtures in \(folder.path)")
            return false
        }
        var results: [(name: String, ok: Bool)] = []
        for fixture in fixtures {
            print("== \(fixture.lastPathComponent)")
            results.append((fixture.lastPathComponent, run(fixture: fixture)))
            Memory.clearCache()
        }
        print("")
        for result in results { print("\(result.ok ? "✓" : "✗") \(result.name)") }
        let ok = results.allSatisfy(\.ok)
        print(ok ? "OK: all \(results.count) fixtures pass" : "FAILED: \(results.filter { !$0.ok }.count) of \(results.count)")
        return ok
    }

    static func run(fixture: URL) -> Bool {
        do {
            let data = try Data(contentsOf: fixture.appending(path: "fixture.json"))
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                print("fixture.json is not an object"); return false
            }
            let references = try loadArrays(url: fixture.appending(path: "references.safetensors"))
            let family = json["family"] as? String ?? "flux2-klein"
            let ok: Bool
            switch family {
            case "flux2-klein": ok = try VerifyKlein.run(fixture: fixture, json: json, references: references)
            case "z-image-turbo": ok = try VerifyZImage.run(fixture: fixture, json: json, references: references)
            case "qwen-image": ok = try VerifyQwenImage.run(fixture: fixture, json: json, references: references)
            case "qwen-image-edit": ok = try VerifyQwenImageEdit.run(fixture: fixture, json: json, references: references)
            case "ming": ok = try VerifyMing.run(fixture: fixture, json: json, references: references)
            case "ltx-2": ok = try VerifyLTX.run(fixture: fixture, json: json, references: references)
            case "seedvr2": ok = try VerifySeedVR2.run(fixture: fixture, json: json, references: references)
            case "sensenova": ok = try VerifySenseNova.run(fixture: fixture, json: json, references: references)
            default:
                print("fixture.json names an unknown family: \(family)"); return false
            }
            let reference = family == "ltx-2" ? "ltx-2-mlx" : family == "sensenova" ? "SenseTime's PyTorch code" : "mflux"
            print(ok ? "OK: the \(family) port matches \(reference) on this checkpoint" : "FAILED: see the stages above")
            return ok
        } catch {
            print("verify failed: \(error)")
            return false
        }
    }

    /// Largest absolute difference over the largest reference magnitude. A stage that does not
    /// `count` is shown with "·" and always passes.
    static func report(_ stage: String, got: MLXArray, want: MLXArray, counts: Bool = true) -> Bool {
        guard got.shape == want.shape else {
            print("  \(stage): shape \(got.shape), expected \(want.shape)")
            return false
        }
        let a = got.asType(.float32)
        let b = want.asType(.float32)
        let difference = abs(a - b).max().item(Float.self)
        let scale = Swift.max(abs(b).max().item(Float.self), 1e-6)
        let relative = difference / scale
        // The typical error, next to the worst one: RMS of the difference over RMS of the reference.
        let rms = sqrt(square(a - b).mean()).item(Float.self) / Swift.max(sqrt(square(b).mean()).item(Float.self), 1e-6)
        let pass = relative <= tolerance
        let mark = !counts ? "·" : pass ? "✓" : "✗"
        print(String(format: "  %@ %-18@ max |Δ| %.5f  (%.3f of the reference's %.4f), RMS %.4f", mark, stage, difference, relative, scale, rms))
        return pass || !counts
    }

    enum VerifyError: LocalizedError {
        case missingReference(String)
        var errorDescription: String? {
            switch self { case .missingReference(let name): "references.safetensors has no \(name)" }
        }
    }
}

/// The FLUX.2 Klein port (Engine/Fixtures/make_klein_fixture.py): the text encoder, one
/// transformer pass, the whole denoising loop with the scheduler, and the VAE decode.
enum VerifyKlein {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let config = KleinConfig.fixture(json)
        print("loading \(fixture.path) as \(config.name)…")
        let model = try KleinModel(modelPath: fixture, config: config, loadTokenizer: false)
        var ok = true

        // 1. Text encoder.
        let encoded = model.encode(inputIds: try reference("input_ids"), attentionMask: try reference("attention_mask"))
        // In bf16, 28 layers carry rounding that depends on the MLX version's kernels (and
        // random weights amplify it): a few percent apart even when every operation is the
        // same. With a float32 reference, that check decides and these lines only inform.
        let float32Reference = references["prompt_embeds_f32"]
        let counts = float32Reference == nil
        ok = Verify.report("text encoder", got: encoded.embeds, want: try reference("prompt_embeds"), counts: counts) && ok
        // Each hidden state on its own: an error that grows with depth is rounding carried
        // through the layers, one that is large from the first is a wrong operation.
        let hidden = config.textEncoder.hiddenSize
        for (index, layer) in config.textEncoderOutLayers.enumerated() {
            let columns = (index * hidden) ..< ((index + 1) * hidden)
            _ = Verify.report("  layer \(layer)", got: encoded.embeds[.ellipsis, columns], want: try reference("prompt_embeds")[.ellipsis, columns], counts: false)
        }
        // The same encoder with float32 activations, against mflux run the same way: rounding
        // then stays at float32's level, so what differs here is the math itself.
        if let wanted = float32Reference {
            let embeds32 = model.textEncoder.promptEmbeds(
                inputIds: try reference("input_ids"), attentionMask: try reference("attention_mask"),
                layers: config.textEncoderOutLayers, computeType: .float32
            )
            eval(embeds32)
            ok = Verify.report("text encoder f32", got: embeds32, want: wanted) && ok
        }
        ok = Verify.report("text ids", got: encoded.ids, want: try reference("text_ids")) && ok

        // 2. Initial noise and ids for the fixture's seed and size.
        let width = (json["width"] as? NSNumber)?.intValue ?? 128
        let height = (json["height"] as? NSNumber)?.intValue ?? 128
        let seed = (json["seed"] as? NSNumber)?.intValue ?? 0
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 4
        let initial = KleinModel.initialLatents(width: width, height: height, seed: seed)
        ok = Verify.report("initial noise", got: initial.latents, want: try reference("latents")) && ok
        ok = Verify.report("latent ids", got: initial.ids, want: try reference("latent_ids")) && ok

        // 3. One transformer pass at the reference's timestep (a value in [0, 1], scaled inside).
        let timestep = try reference("timestep").asType(.float32).item(Float.self)
        let noise = model.transformer(latents: try reference("latents"), prompt: try reference("prompt_embeds"), timestep: timestep, imageIds: try reference("latent_ids"), textIds: try reference("text_ids"))
        eval(noise)
        ok = Verify.report("transformer pass", got: noise, want: try reference("noise")) && ok
        ok = try VerifyLoRA.run(fixture: fixture, model: model, plain: noise) {
            model.transformer(latents: try reference("latents"), prompt: try reference("prompt_embeds"), timestep: timestep,
                              imageIds: try reference("latent_ids"), textIds: try reference("text_ids"))
        } && ok

        // 4. The scheduler (shifted for the image's token count, as Klein runs it) and the loop.
        let schedule = FlowMatchSchedule(steps: steps, imageSeqLen: initial.latentHeight * initial.latentWidth)
        ok = Verify.report("sigmas", got: MLXArray(schedule.sigmas), want: try reference("sigmas")) && ok
        ok = Verify.report("timesteps", got: MLXArray(schedule.timesteps), want: try reference("timesteps")) && ok
        var latents = try reference("latents")
        let prompt = try reference("prompt_embeds")
        for t in 0 ..< steps {
            let predicted = model.transformer(latents: latents, prompt: prompt, timestep: schedule.timesteps[t], imageIds: initial.ids, textIds: encoded.ids)
            latents = KleinModel.step(latents: latents, noise: predicted, schedule: schedule, index: t)
            eval(latents)
        }
        ok = Verify.report("denoised latents", got: latents, want: try reference("final_latents")) && ok

        // 5. VAE decode (the reference is channels first).
        let grid = try reference("final_latents").reshaped([1, initial.latentHeight, initial.latentWidth, 128])
        let decoded = model.vae.decodePacked(grid)
        eval(decoded)
        let wanted = try reference("decoded").transposed(0, 2, 3, 1)
        ok = Verify.report("vae decode", got: decoded, want: wanted) && ok

        // 6. An edit from one picture, as Flux2KleinEdit runs it (fixtures made before it have none).
        if let picture = references["ref_image"] {
            ok = try verifyEdit(model: model, picture: picture.transposed(0, 2, 3, 1), json: json, reference: reference,
                                steps: steps, schedule: schedule, latentHeight: initial.latentHeight, latentWidth: initial.latentWidth) && ok
        }
        return ok
    }

    /// The VAE encoder, the picture's tokens and ids, one pass and the whole loop with them, the
    /// decode; and the sizes pictures are encoded at.
    static func verifyEdit(
        model: KleinModel, picture: MLXArray, json: [String: Any], reference: (String) throws -> MLXArray,
        steps: Int, schedule: FlowMatchSchedule, latentHeight: Int, latentWidth: Int
    ) throws -> Bool {
        var ok = true
        let encoded = model.vae.encode(picture)
        eval(encoded)
        ok = Verify.report("vae encode", got: encoded, want: try reference("ref_encoded").transposed(0, 2, 3, 1)) && ok
        let ref = model.referenceTokens(pixels: picture)
        ok = Verify.report("reference tokens", got: ref.tokens, want: try reference("ref_tokens")) && ok
        ok = Verify.report("reference ids", got: ref.ids, want: try reference("ref_ids")) && ok

        // One pass from the reference's own inputs, then the loop from the port's.
        let count = latentHeight * latentWidth
        let prompt = try reference("prompt_embeds")
        let textIds = try reference("text_ids")
        let ids = concatenated([try reference("latent_ids"), try reference("ref_ids")], axis: 1)
        let timestep = try reference("timestep").asType(.float32).item(Float.self)
        let input = concatenated([try reference("latents"), try reference("ref_tokens")], axis: 1)
        let noise = model.transformer(latents: input, prompt: prompt, timestep: timestep, imageIds: ids, textIds: textIds)[0..., 0 ..< count, 0...]
        eval(noise)
        ok = Verify.report("edit pass", got: noise, want: try reference("edit_noise")) && ok
        var latents = try reference("latents")
        let portIds = concatenated([KleinModel.initialLatents(width: 16 * latentWidth, height: 16 * latentHeight, seed: 0).ids, ref.ids], axis: 1)
        for t in 0 ..< steps {
            let predicted = model.transformer(
                latents: concatenated([latents, ref.tokens], axis: 1), prompt: prompt, timestep: schedule.timesteps[t],
                imageIds: portIds, textIds: textIds
            )[0..., 0 ..< count, 0...]
            latents = KleinModel.step(latents: latents, noise: predicted, schedule: schedule, index: t)
            eval(latents)
        }
        ok = Verify.report("edit latents", got: latents, want: try reference("edit_final_latents")) && ok
        let decoded = model.vae.decodePacked(try reference("edit_final_latents").reshaped([1, latentHeight, latentWidth, 128]))
        eval(decoded)
        ok = Verify.report("edit decode", got: decoded, want: try reference("edit_decoded").transposed(0, 2, 3, 1)) && ok

        // `reference_dims`: scaled to about a megapixel, rounded half to even, cut to multiples of 16.
        for case let row as [NSNumber] in json["reference_sizes"] as? [Any] ?? [] where row.count == 4 {
            let (width, height) = (row[0].intValue, row[1].intValue)
            let size = KleinReference.sizes(width: width, height: height).encoded
            let pass = size.width == row[2].intValue && size.height == row[3].intValue
            print("  \(pass ? "✓" : "✗") reference size    \(width) × \(height) → \(size.width) × \(size.height)"
                  + (pass ? "" : ", expected \(row[2]) × \(row[3])"))
            ok = pass && ok
        }
        return ok
    }
}

/// The LoRA files Engine/Fixtures/make_lora_fixture.py (or make_ltx_fixture.py --lora) added to a
/// fixture (`lora/`), each put on the port's transformer for the fixture's pass, against the
/// reference's pass with the same file (`<output>_<format>` for each of the pass's outputs); then
/// taken off, which must give the plain pass back exactly. A fixture without them passes.
enum VerifyLoRA {
    static func run(fixture: URL, model: LoRAAdaptable, plain: MLXArray, pass: () throws -> MLXArray) throws -> Bool {
        try run(fixture: fixture, model: model, plain: ["noise": plain]) { ["noise": try pass()] }
    }

    static func run(fixture: URL, model: LoRAAdaptable, plain: [String: MLXArray], pass: () throws -> [String: MLXArray]) throws -> Bool {
        let folder = fixture.appending(path: "lora", directoryHint: .isDirectory)
        guard let data = try? Data(contentsOf: folder.appending(path: "lora.json")),
              let entries = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return true }
        let references = try loadArrays(url: folder.appending(path: "references.safetensors"))
        var ok = true
        for entry in entries {
            guard let format = entry["format"] as? String, let file = entry["file"] as? String,
                  let scale = (entry["scale"] as? NSNumber)?.doubleValue
            else { continue }
            for line in try model.loras.set([LoRASpec(path: folder.appending(path: file).path, scale: scale)], on: model.adaptedModule) {
                print("  " + line.replacingOccurrences(of: "[turbo] ", with: ""))
            }
            let outputs = try pass()
            for (name, output) in outputs.sorted(by: { $0.key > $1.key }) {
                guard let wanted = references["\(name)_\(format)"] else { throw Verify.VerifyError.missingReference("lora/\(name)_\(format)") }
                eval(output)
                let label = outputs.count == 1 ? "LoRA \(format)" : "LoRA \(format) \(name)"
                ok = Verify.report(label, got: output, want: wanted) && ok
                // What the reference itself makes of the file when it bakes it (shown).
                if let baked = references["\(name)_\(format)_baked"] {
                    _ = Verify.report("  baked by the reference", got: output, want: baked, counts: false)
                }
            }
        }
        try model.loras.set([], on: model.adaptedModule)
        let again = try pass()
        let restored = plain.allSatisfy { name, array in
            guard let output = again[name] else { return false }
            eval(output)
            return (output .== array).all().item(Bool.self)
        }
        print("  \(restored ? "✓" : "✗") LoRAs taken off    \(restored ? "the plain pass again, bit for bit" : "differs from the plain pass")")
        return restored && ok
    }
}
