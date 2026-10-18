import Foundation
import MLX
import TurboEngineCore

/// The FLUX 3 Image Klein port (Engine/Fixtures/make_flux3_fixture.py): the checkpoint read from
/// BFL's layout and quantized on loading, the prompt, the noise and ids, one pass, the schedule and
/// the loop, the decode; an edit from two pictures (their preparation byte for byte, their tokens,
/// a pass, the loop, the decode) and the padding of a single one; the Base's guided pass.
enum VerifyFlux3 {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let config = Flux3Config.fixture(json)
        let bits = (json["bits"] as? NSNumber)?.intValue
        print("loading \(fixture.path) as \(config.name), quantized to \(bits.map { "\($0) bits" } ?? "nothing")…")
        let model = try Flux3Model(modelPath: fixture, config: config, quantize: bits, loadTokenizer: false)
        var ok = true

        // 1. The prompt: the padding's positions, the encoder in bf16 (shown) and in float32 (decides).
        let mask = try reference("attention_mask")
        let encoded = try model.encode(inputIds: try reference("input_ids"), attentionMask: mask)
        ok = Verify.report("text positions", got: Flux3Model.positions(attentionMask: mask), want: try reference("positions")) && ok
        _ = Verify.report("text encoder", got: encoded.embeds, want: try reference("prompt_embeds"), counts: false)
        let embeds32 = try model.encode(inputIds: try reference("input_ids"), attentionMask: mask, computeType: .float32).embeds
        ok = Verify.report("text encoder f32", got: embeds32, want: try reference("prompt_embeds_f32")) && ok
        // Each hidden state on its own: an error that grows with depth is rounding carried through
        // the layers, one that is large from the first is a wrong operation.
        let hidden = config.textEncoder.hiddenSize
        for (index, layer) in config.textEncoderOutLayers.enumerated() {
            let columns = (index * hidden) ..< ((index + 1) * hidden)
            _ = Verify.report("  layer \(layer)", got: embeds32[.ellipsis, columns], want: try reference("prompt_embeds_f32")[.ellipsis, columns], counts: false)
        }
        let realCount = mask.sum().item(Int.self)
        _ = Verify.report("  real tokens", got: embeds32[0..., 0 ..< realCount], want: try reference("prompt_embeds_f32")[0..., 0 ..< realCount], counts: false)
        _ = Verify.report("  padding", got: embeds32[0..., realCount...], want: try reference("prompt_embeds_f32")[0..., realCount...], counts: false)
        ok = Verify.report("text ids", got: encoded.ids, want: try reference("text_ids")) && ok
        if let emptyIds = references["empty_ids"], let emptyMask = references["empty_mask"] {
            let empty = try model.encode(inputIds: emptyIds, attentionMask: emptyMask)
            _ = Verify.report("empty prompt", got: empty.embeds, want: try reference("empty_embeds"), counts: false)
        }

        // 2. Noise and ids for the fixture's seed and size.
        let (width, height, seed, steps) = (int("width", 96), int("height", 128), int("seed", 7), int("steps", 4))
        let (latentHeight, latentWidth) = (height / 16, width / 16)
        ok = Verify.report("initial noise", got: Flux3Model.initialNoise(width: width, height: height, seed: seed), want: try reference("latents")) && ok
        let ids = Flux3Model.imageIds(height: latentHeight, width: latentWidth)
        ok = Verify.report("latent ids", got: ids, want: try reference("latent_ids")) && ok

        // 3. The schedule, one pass at its second step (from the reference's inputs), the loop.
        let schedule = Flux3Schedule(steps: steps)
        ok = Verify.report("timesteps", got: MLXArray(schedule.timesteps), want: try reference("timesteps")) && ok
        let prompt = EncodedPrompt(embeds: try reference("prompt_embeds"), ids: try reference("text_ids"))
        let latents = try reference("latents")
        let rotary = model.transformer.rotary(imageIds: try reference("latent_ids"), referenceIds: nil, textIds: prompt.ids)
        func pass(_ x: MLXArray, index: Int, references tokens: MLXArray? = nil, rotary: Flux3Rotary) -> MLXArray {
            model.velocity(latents: x, prompt: prompt, rotary: rotary, unconditional: nil, guidance: 1, references: tokens,
                           schedule: schedule, index: index)
        }
        let noise = pass(latents, index: 1, rotary: rotary)
        eval(noise)
        ok = Verify.report("transformer pass", got: noise, want: try reference("noise")) && ok
        ok = try VerifyLoRA.run(fixture: fixture, model: model, plain: noise) { pass(latents, index: 1, rotary: rotary) } && ok
        for index in 0 ..< steps {
            guard let image = references["time_image_\(index)"], let context = references["time_context_\(index)"] else { break }
            let time = model.transformer.timeEmbeddings(timestep: schedule.timesteps[index], targetTimestep: schedule.timesteps[index + 1])
            eval(time.image, time.context)
            ok = Verify.report("time \(index + 1) image", got: time.image, want: image) && ok
            ok = Verify.report("time \(index + 1) context", got: time.context, want: context) && ok
        }
        if let first = references["first_noise"] {
            let firstNoise = pass(latents, index: 0, rotary: rotary)
            eval(firstNoise)
            ok = Verify.report("first pass", got: firstNoise, want: first) && ok
        }
        if let last = references["last_noise"], let before = references["latents_\(steps - 1)"] {
            let lastNoise = pass(before, index: steps - 1, rotary: rotary)
            eval(lastNoise)
            ok = Verify.report("last pass", got: lastNoise, want: last) && ok
            let stepped = Flux3Model.step(latents: before, velocity: last, schedule: schedule, index: steps - 1)
            _ = Verify.report("  last update", got: stepped, want: try reference("final_latents"), counts: false)
        }
        var x = latents
        for t in 0 ..< steps {
            // Each step from the reference's own state, where the fixture has it: a step that
            // drifts is then told from drift carried over from the steps before.
            let start = t > 0 ? references["latents_\(t)"] ?? x : x
            let next = Flux3Model.step(latents: start, velocity: pass(start, index: t, rotary: rotary), schedule: schedule, index: t)
            eval(next)
            if let wanted = references["latents_\(t + 1)"] {
                _ = Verify.report("  step \(t + 1)", got: next, want: wanted, counts: false)
            }
            x = Flux3Model.step(latents: x, velocity: pass(x, index: t, rotary: rotary), schedule: schedule, index: t)
            eval(x)
        }
        ok = Verify.report("denoised latents", got: x, want: try reference("final_latents")) && ok
        let decoded = model.vae.decodePacked(try reference("final_latents").reshaped([1, latentHeight, latentWidth, 128]))
        eval(decoded)
        ok = Verify.report("vae decode", got: decoded, want: try reference("decoded").transposed(0, 2, 3, 1)) && ok

        // A check of the real checkpoint records text to image only.
        guard FileManager.default.fileExists(atPath: fixture.appending(path: "picture1.png").path) else { return ok }

        // 4. An edit from two pictures: each prepared as one of several, byte for byte.
        let pictures = try (1 ... 2).map { try Flux3References.load(fixture.appending(path: "picture\($0).png").path) }
        let prepared = try Flux3References.prepare(pictures)
        for (index, picture) in prepared.pictures.enumerated() {
            ok = bytes("picture \(index + 1)", picture, want: try reference("multi\(index + 1)")) && ok
        }
        let cond = model.referenceTokens(prepared.pictures)
        // The VAE's encoder carries bf16 rounding in both engines (Klein's check: 1.6%, 1.1% RMS);
        // the small picture decides, the megapixel one only shows the largest error growing with
        // the pixels while the RMS stays put.
        let firstCount = (prepared.pictures[0].width / 16) * (prepared.pictures[0].height / 16)
        ok = Verify.report("reference tokens 1", got: cond.tokens[0..., 0 ..< firstCount], want: try reference("cond_tokens")[0..., 0 ..< firstCount]) && ok
        _ = Verify.report("reference tokens 2", got: cond.tokens[0..., firstCount...], want: try reference("cond_tokens")[0..., firstCount...], counts: false)
        ok = Verify.report("reference ids", got: cond.ids, want: try reference("cond_ids")) && ok
        let editRotary = model.transformer.rotary(imageIds: ids, referenceIds: cond.ids, textIds: prompt.ids)
        let editNoise = pass(latents, index: 1, references: try reference("cond_tokens"), rotary: editRotary)
        eval(editNoise)
        ok = Verify.report("edit pass", got: editNoise, want: try reference("edit_noise")) && ok
        x = latents
        for t in 0 ..< steps {
            x = Flux3Model.step(latents: x, velocity: pass(x, index: t, references: cond.tokens, rotary: editRotary), schedule: schedule, index: t)
            eval(x)
        }
        ok = Verify.report("edit latents", got: x, want: try reference("edit_final_latents")) && ok
        let editDecoded = model.vae.decodePacked(try reference("edit_final_latents").reshaped([1, latentHeight, latentWidth, 128]))
        eval(editDecoded)
        ok = Verify.report("edit decode", got: editDecoded, want: try reference("edit_decoded").transposed(0, 2, 3, 1)) && ok

        // 5. A single picture, padded by reflection to multiples of 16, and the box it covers.
        let single = try Flux3References.prepare([try Flux3References.load(fixture.appending(path: "picture3.png").path)])
        ok = bytes("padded picture", single.pictures[0], want: try reference("single")) && ok
        let wantedBox = (json["single_box"] as? [NSNumber])?.map(\.intValue) ?? []
        let box = single.box.map { [$0.left, $0.top, $0.right, $0.bottom] } ?? []
        print("  \(box == wantedBox ? "✓" : "✗") padded box         \(box)" + (box == wantedBox ? "" : ", expected \(wantedBox)"))
        ok = box == wantedBox && ok

        // 6. The sizes pictures are prepared at.
        for case let row as [NSNumber] in json["canvas_sizes"] as? [Any] ?? [] where row.count == 6 {
            let (w, h) = (row[0].intValue, row[1].intValue)
            let sizes = Flux3References.referenceCanvas(width: w, height: h)
            let got = [sizes.fit.width, sizes.fit.height, sizes.canvas.width, sizes.canvas.height]
            let wanted = row[2...].map(\.intValue)
            print("  \(got == wanted ? "✓" : "✗") one picture       \(w) × \(h) → \(got[0]) × \(got[1]) on \(got[2]) × \(got[3])"
                  + (got == wanted ? "" : ", expected \(wanted)"))
            ok = got == wanted && ok
        }
        for case let row as [NSNumber] in json["multi_sizes"] as? [Any] ?? [] where row.count == 4 {
            let (w, h) = (row[0].intValue, row[1].intValue)
            let cut = Flux3References.multiReferenceSizes(width: w, height: h).cut
            let pass = cut.width == row[2].intValue && cut.height == row[3].intValue
            print("  \(pass ? "✓" : "✗") one of several    \(w) × \(h) → \(cut.width) × \(cut.height)" + (pass ? "" : ", expected \(row[2]) × \(row[3])"))
            ok = pass && ok
        }

        // 7. The Base: the same checkpoint without its target time, guided against the empty prompt.
        let base = try Flux3Model(modelPath: fixture, config: config, quantize: bits, flowMap: false, loadTokenizer: false)
        let emptyPrompt = EncodedPrompt(embeds: try reference("empty_embeds"), ids: Flux3Model.textIds(count: try reference("empty_embeds").shape[1]))
        let guidance = (json["guidance"] as? NSNumber)?.doubleValue ?? 3.5
        let baseNoise = base.velocity(
            latents: latents, prompt: prompt, rotary: rotary,
            unconditional: (emptyPrompt, base.transformer.rotary(imageIds: try reference("latent_ids"), referenceIds: nil, textIds: emptyPrompt.ids)),
            guidance: guidance, references: nil, schedule: schedule, index: 1
        )
        eval(baseNoise)
        ok = Verify.report("base guided pass", got: baseNoise, want: try reference("base_noise")) && ok
        return ok
    }

    /// A prepared picture against the reference's bytes [H, W, 3], which must be the same exactly.
    static func bytes(_ stage: String, _ picture: Flux3References.Picture, want: MLXArray) -> Bool {
        guard want.shape == [picture.height, picture.width, 3] else {
            print("  ✗ \(stage): \(picture.width) × \(picture.height), expected \(want.shape[1]) × \(want.shape[0])")
            return false
        }
        let wanted = want.asType(.uint8).asArray(UInt8.self)
        let differing = zip(picture.rgb, wanted).filter { $0 != $1 }.count
        let largest = zip(picture.rgb, wanted).map { abs(Int($0) - Int($1)) }.max() ?? 0
        print("  \(differing == 0 ? "✓" : "✗") \(stage.padding(toLength: 18, withPad: " ", startingAt: 0)) \(picture.width) × \(picture.height), "
              + (differing == 0 ? "the same bytes" : "\(differing) bytes differ, by up to \(largest)"))
        return differing == 0
    }
}
