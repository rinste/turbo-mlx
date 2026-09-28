import Foundation
import MLX
import TurboEngineCore

/// The Qwen-Image-Edit port against mflux's outputs on a small checkpoint
/// (Engine/Fixtures/make_qwen_image_edit_fixture.py): the picture's two resizes and patches, the
/// vision tower, the prompt with the picture's tokens, the picture's latents, one transformer pass
/// over the image and the picture, the guided loop and the decode.
enum VerifyQwenImageEdit {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let config = QwenImageConfig.fixture(json)
        guard let vision = config.vision else { print("fixture.json has no vision tower"); return false }
        print("loading \(fixture.path) as \(config.name) with a vision tower…")
        let model = try QwenImageEditModel(modelPath: fixture, config: config, loadTokenizer: false)
        var ok = true

        // 1. The picture as the vision tower reads it: Pillow's two bicubic resizes (exact bytes),
        // the normalized patches and their grid.
        let rgb = try reference("picture_rgb")
        let picture = QwenEditPicture(rgb: rgb.asArray(UInt8.self), width: rgb.shape[1], height: rgb.shape[0])
        let input = picture.visionInput(vision)
        let resized = MLXArray(input.resized.rgb, [input.resized.height, input.resized.width, 3])
        ok = Verify.report("bicubic resizes", got: resized, want: try reference("vl_rgb")) && ok
        ok = Verify.report("vision patches", got: input.pixelValues, want: try reference("pixel_values")) && ok
        let grid = try reference("image_grid_thw").asArray(Int32.self).map { Int($0) }
        if grid != [input.grid.t, input.grid.h, input.grid.w] {
            print("  ✗ vision grid        \([input.grid.t, input.grid.h, input.grid.w]), expected \(grid)")
            ok = false
        }

        // 2. The vision tower on the reference's patches.
        let encoder = try model.loadedTextEncoder()
        let grids = [QwenVisionGrid(t: grid[0], h: grid[1], w: grid[2])]
        let image = encoder.imageEmbeds(pixelValues: try reference("pixel_values"), grids: grids)
        eval(image)
        ok = Verify.report("vision tower", got: image, want: try reference("image_embeds")) && ok

        // 3. The prompts with the picture's tokens in place (the reference's, to isolate this stage).
        let pictureTokens = try reference("image_embeds")
        let prompt = encoder.editEmbeds(inputIds: try reference("input_ids"), imageEmbeds: pictureTokens)
        ok = Verify.report("edit prompt", got: prompt, want: try reference("prompt_embeds")) && ok
        let negative = encoder.editEmbeds(inputIds: try reference("negative_input_ids"), imageEmbeds: pictureTokens)
        ok = Verify.report("negative prompt", got: negative, want: try reference("negative_prompt_embeds")) && ok

        // 4. The picture at the image's size (Pillow's Lanczos), then its latents.
        let width = (json["width"] as? NSNumber)?.intValue ?? 96
        let height = (json["height"] as? NSNumber)?.intValue ?? 64
        let size = QwenImageEditModel.referenceSize(pictureWidth: picture.width, pictureHeight: picture.height, width: width, height: height)
        if size.width != width || size.height != height {
            print("  ✗ reference size     \(size), expected the image's \(width) × \(height)")
            ok = false
        }
        let vaeInput = picture.vaeInput(width: width, height: height)
        let lanczos = ((vaeInput + 1) / 2 * 255).round().asType(.uint8)[0]
        ok = Verify.report("lanczos resize", got: lanczos, want: try reference("vae_rgb")) && ok
        ok = Verify.report("vae input", got: vaeInput, want: try reference("vae_input").transposed(0, 2, 3, 1)) && ok
        let (transformer, vae) = try model.loadedImageSide()
        let referenceLatents = model.referenceLatents(picture: picture, width: width, height: height, vae: vae)
        ok = Verify.report("picture latents", got: referenceLatents, want: try reference("reference_latents")) && ok

        // 5. One pass over the image and the picture, the image's part kept.
        let latentGrids = [(height: height / 16, width: width / 16), (height: height / 16, width: width / 16)]
        let latents = try reference("latents")
        let count = latents.shape[1]
        let pictureLatents = try reference("reference_latents")
        let timestep = try reference("timestep").asType(.float32).item(Float.self)
        let embeds = try reference("prompt_embeds")
        let noise = transformer(latents: concatenated([latents, pictureLatents], axis: 1), prompt: embeds, timestep: timestep,
                                grids: latentGrids)[0..., 0 ..< count]
        eval(noise)
        ok = Verify.report("transformer pass", got: noise, want: try reference("noise")) && ok

        // 6. The schedule and the guided loop.
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 3
        let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 2.5
        let schedule = LinearSchedule(steps: steps, width: width, height: height, shift: config.shift)
        ok = Verify.report("sigmas", got: MLXArray(schedule.sigmas), want: try reference("sigmas")) && ok
        let negativeEmbeds = try reference("negative_prompt_embeds")
        var x = latents
        for t in 0 ..< steps {
            let input = concatenated([x, pictureLatents], axis: 1)
            let positive = transformer(latents: input, prompt: embeds, timestep: schedule.sigmas[t], grids: latentGrids)[0..., 0 ..< count]
            let unconditional = transformer(latents: input, prompt: negativeEmbeds, timestep: schedule.sigmas[t], grids: latentGrids)[0..., 0 ..< count]
            x = schedule.step(latents: x, noise: QwenImageModel.guidedNoise(positive, negative: unconditional, guidance: guidance), index: t)
            eval(x)
        }
        ok = Verify.report("denoised latents", got: x, want: try reference("final_latents")) && ok

        // 7. The decode (the reference is channels first).
        let grid2d = QwenImageModel.unpack(try reference("final_latents"), latentHeight: height / 16, latentWidth: width / 16)
        let decoded = vae.decode(grid2d)
        eval(decoded)
        ok = Verify.report("vae decode", got: decoded, want: try reference("decoded").transposed(0, 2, 3, 1)) && ok
        return ok
    }
}
