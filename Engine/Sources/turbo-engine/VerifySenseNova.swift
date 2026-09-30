import Foundation
import MLX
import TurboEngineCore

/// The SenseNova-U1.5 port (Engine/Fixtures/make_sensenova_fixture.py, SenseTime's PyTorch code in
/// float32): the prompt caches layer by layer, the schedule, the generation embedding and the model's
/// input at the first step, both velocities of the first step, then the whole loop from the
/// reference's noise, with guidance and without; then an edit from a picture. A recording of the reference on a real pack
/// (Engine/Reference/sensenova_reference.py --record) names the pack in `fixture.json` and has
/// only the stages of its own run (in bf16, where rounding alone moves the later ones).
enum VerifySenseNova {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let modelPath = (json["pack"] as? String).map { URL(fileURLWithPath: $0) } ?? fixture
        print("loading \(modelPath.path)…")
        let model = try SenseNovaModel(modelPath: modelPath, loadTokenizer: false)
        // The reference's own weights: the pack's dequantized into bf16.
        if json["dequantize"] as? Bool == true { try model.dequantizeForParity() }
        let config = model.config
        model.timestepShift = (json["timestep_shift"] as? NSNumber)?.doubleValue ?? 3
        model.tEps = (json["t_eps"] as? NSNumber)?.floatValue ?? 0.02
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 3
        let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 2
        let dtype = try model.activationType()
        var ok = true

        // 1. The prompt caches, from the reference's token ids.
        func prefix(_ tag: String) throws -> SenseNovaModel.Prefix {
            let ids = try reference("\(tag)_input_ids").asType(.int32)
            let port = try model.prefix(ids: ids.reshaped([-1]).asArray(Int32.self).map(Int.init))
            for layer in 0 ..< config.numLayers {
                ok = Verify.report("\(tag) keys \(layer)", got: port.cache[layer].0, want: try reference("\(tag)_keys_\(layer)")) && ok
                ok = Verify.report("\(tag) values \(layer)", got: port.cache[layer].1, want: try reference("\(tag)_values_\(layer)")) && ok
            }
            return port
        }
        let conditional = try prefix("cond")
        let hasUnconditional = references["uncond_input_ids"] != nil
        let unconditional = hasUnconditional ? try prefix("uncond") : nil
        // The reference's own caches, for the stages below.
        func referencePrefix(_ tag: String) throws -> SenseNovaModel.Prefix {
            let cache = try (0 ..< config.numLayers).map {
                (try reference("\(tag)_keys_\($0)").asType(dtype), try reference("\(tag)_values_\($0)").asType(dtype))
            }
            return SenseNovaModel.Prefix(cache: cache, position: cache[0].0.shape[2])
        }
        let wantedConditional = try referencePrefix("cond")

        // 2. The schedule.
        let timesteps = SenseNovaConfig.timesteps(steps: steps, shift: model.timestepShift)
        ok = Verify.report("timesteps", got: MLXArray(timesteps), want: try reference("timesteps")) && ok

        // 3. The first step: the image's embedding, the model's input, both velocities.
        let noise = try reference("noise").transposed(0, 2, 3, 1)
        let tokens = (noise.shape[1] / config.tokenSize) * (noise.shape[2] / config.tokenSize)
        let noiseScale = config.noiseScale(tokens: tokens)
        let image = noise.asType(dtype) * Float(noiseScale)
        let vision = try model.imageEmbeds(image, t: timesteps[0], noiseScale: noiseScale)
        eval(vision)
        ok = Verify.report("image embeds", got: vision[0], want: try reference("image_embeds")[0]) && ok
        let embeds = try reference("image_embeds").asType(dtype)
        let conditionalVelocity = try model.velocity(embeds: embeds, image: image, prefix: wantedConditional, t: timesteps[0])
        ok = Verify.report("velocity", got: patchified(conditionalVelocity, size: config.tokenSize), want: try reference("v_cond")) && ok
        if hasUnconditional {
            let unconditionalVelocity = try model.velocity(embeds: embeds, image: image, prefix: try referencePrefix("uncond"), t: timesteps[0])
            ok = Verify.report("velocity uncond", got: patchified(unconditionalVelocity, size: config.tokenSize), want: try reference("v_uncond")) && ok
        }

        // Each step of a recording from the reference's own image and input: the velocity's error
        // step by step (the rest of the difference is the two trajectories drifting apart).
        for index in 0 ..< steps {
            guard let z = references["step_\(index)_z"], let stepEmbeds = references["step_\(index)_embeds"] else { break }
            let (rows, columns) = (image.shape[1] / config.tokenSize, image.shape[2] / config.tokenSize)
            let size = config.tokenSize
            let stepImage = z.reshaped([1, rows, columns, size, size, 3]).transposed(0, 1, 3, 2, 4, 5)
                .reshaped([1, rows * size, columns * size, 3]).asType(dtype)
            let velocity = try model.velocity(embeds: stepEmbeds.asType(dtype), image: stepImage, prefix: wantedConditional, t: timesteps[index])
            _ = Verify.report("velocity \(index)", got: patchified(velocity, size: size), want: try reference("step_\(index)_v"), counts: false)
            let port = try model.imageEmbeds(stepImage, t: timesteps[index], noiseScale: noiseScale)
            _ = Verify.report("embeds \(index)", got: port, want: stepEmbeds, counts: false)
        }

        // 4. The loops, from the port's own caches.
        let denoised = try model.denoise(noise: noise, prefix: conditional, unconditional: unconditional, steps: steps, guidance: guidance)
        ok = Verify.report("image", got: denoised, want: try reference("image").transposed(0, 2, 3, 1)) && ok
        // A recording's loop again from the reference's own prompt cache: what the cache's rounding adds.
        if json["pack"] != nil {
            let fromReference = try model.denoise(noise: noise, prefix: wantedConditional, unconditional: nil, steps: steps, guidance: 1)
            _ = Verify.report("image, its cache", got: fromReference, want: try reference("image").transposed(0, 2, 3, 1), counts: false)
        }
        if references["image_no_guidance"] != nil {
            let unguided = try model.denoise(noise: noise, prefix: conditional, unconditional: nil, steps: steps, guidance: 1)
            ok = Verify.report("image, guidance 1", got: unguided, want: try reference("image_no_guidance").transposed(0, 2, 3, 1)) && ok
        }
        if references["edit_image"] != nil {
            ok = try verifyEdit(model: model, fixture: fixture, json: json, reference: reference, noise: noise, steps: steps, guidance: guidance) && ok
        }
        return ok
    }

    /// An edit of `picture.png`: its preparation byte for byte, its pixels and patch embedding, the
    /// query's positions, both prefixes (the picture-only one is guidance's unconditional query),
    /// the first step's velocities and the loop.
    static func verifyEdit(
        model: SenseNovaModel, fixture: URL, json: [String: Any], reference: (String) throws -> MLXArray,
        noise: MLXArray, steps: Int, guidance: Float
    ) throws -> Bool {
        var ok = true
        let config = model.config
        let dtype = try model.activationType()
        let width = (json["width"] as? NSNumber)?.intValue ?? 128
        let height = (json["height"] as? NSNumber)?.intValue ?? 96
        let imageStart = (json["image_start"] as? NSNumber)?.intValue ?? 151_670
        let imageContext = (json["image_context"] as? NSNumber)?.intValue ?? 151_669

        let picture = SenseNovaPicture(try QwenEditPicture(path: fixture.appending(path: "picture.png").path), area: width * height)
        let wantedPicture = try reference("edit_picture")
        let bytes = MLXArray(picture.rgb, [picture.height, picture.width, 3])
        ok = Verify.report("edit picture", got: bytes.asType(.float32), want: wantedPicture.asType(.float32)) && ok
        let pixels = picture.pixelValues
        ok = Verify.report("edit pixels", got: pixels, want: try reference("edit_pixels")) && ok
        let features = try model.pictureFeatures(pixels)
        ok = Verify.report("edit features", got: features[0], want: try reference("edit_features")) && ok

        let rows = picture.height / config.tokenSize
        let columns = picture.width / config.tokenSize
        func prefix(_ tag: String) throws -> SenseNovaModel.Prefix {
            let ids = try reference("\(tag)_input_ids").reshaped([-1]).asArray(Int32.self).map(Int.init)
            let positions = SenseNovaPositions.query(ids: ids, imageStart: imageStart, imageContext: imageContext, grids: [(rows, columns)])
            let port = MLXArray(positions.t + positions.h + positions.w, [3, positions.count])
            ok = Verify.report("\(tag) positions", got: port.asType(.float32), want: try reference("\(tag)_indexes").asType(.float32)) && ok
            let prefix = try model.prefix(ids: ids, pictures: [pixels], imageStart: imageStart, imageContext: imageContext)
            for layer in 0 ..< config.numLayers {
                ok = Verify.report("\(tag) keys \(layer)", got: prefix.cache[layer].0, want: try reference("\(tag)_keys_\(layer)")) && ok
                ok = Verify.report("\(tag) values \(layer)", got: prefix.cache[layer].1, want: try reference("\(tag)_values_\(layer)")) && ok
            }
            let wantedPosition = Int(try reference("\(tag)_indexes")[0].max().item(Int32.self)) + 1
            if prefix.position != wantedPosition {
                print("  ✗ \(tag) image position \(prefix.position), expected \(wantedPosition)")
                ok = false
            }
            return prefix
        }
        let conditional = try prefix("edit_cond")
        let unconditional = try prefix("edit_uncond")

        let timesteps = SenseNovaConfig.timesteps(steps: steps, shift: model.timestepShift)
        let tokens = (noise.shape[1] / config.tokenSize) * (noise.shape[2] / config.tokenSize)
        let image = noise.asType(dtype) * Float(config.noiseScale(tokens: tokens))
        let embeds = try reference("edit_image_embeds").asType(dtype)
        let velocity = try model.velocity(embeds: embeds, image: image, prefix: conditional, t: timesteps[0])
        ok = Verify.report("edit velocity", got: patchified(velocity, size: config.tokenSize), want: try reference("edit_v_cond")) && ok
        let other = try model.velocity(embeds: embeds, image: image, prefix: unconditional, t: timesteps[0])
        ok = Verify.report("edit velocity uncond", got: patchified(other, size: config.tokenSize), want: try reference("edit_v_uncond")) && ok

        let edited = try model.denoise(noise: noise, prefix: conditional, unconditional: unconditional, steps: steps, guidance: guidance)
        ok = Verify.report("edit image", got: edited, want: try reference("edit_image").transposed(0, 2, 3, 1)) && ok
        return ok
    }

    /// [1, H, W, 3] → the reference's [1, tokens, size · size · 3] (`patchify`, channels last).
    static func patchified(_ image: MLXArray, size: Int) -> MLXArray {
        let (rows, columns) = (image.shape[1] / size, image.shape[2] / size)
        return image.reshaped([1, rows, size, columns, size, 3]).transposed(0, 1, 3, 2, 4, 5).reshaped([1, rows * columns, size * size * 3])
    }
}
