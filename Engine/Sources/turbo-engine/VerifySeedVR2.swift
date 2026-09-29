import Foundation
import MLX
import TurboEngineCore

/// The SeedVR2 port against mflux's outputs on a small checkpoint
/// (Engine/Fixtures/make_seedvr2_fixture.py): the picture's preparation, the tiled encode, the
/// noise, one transformer pass, the step, the tiled decode, the color correction, and the whole
/// upscale from the picture.
enum VerifySeedVR2 {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let config = SeedVR2Config.fixture(json)
        print("loading \(fixture.path) as \(config.name)…")
        let model = try SeedVR2Model(modelPath: fixture, config: config, textEmbedding: try reference("text"))
        var ok = true

        // 1. The picture: the output size, Pillow's bicubic, the padding (with and without softness).
        let rgb = try reference("picture_rgb")
        let picture = SeedVR2Picture(rgb: rgb.asArray(UInt8.self), width: rgb.shape[1], height: rgb.shape[0])
        let factor = (json["upscale"] as? NSNumber)?.doubleValue ?? 2
        let size = SeedVR2Picture.outputSize(width: picture.width, height: picture.height, factor: factor)
        let expected = (json["output"] as? [NSNumber])?.map(\.intValue) ?? []
        let sizeOK = expected == [size.width, size.height]
        print("  \(sizeOK ? "✓" : "✗") output size        \(picture.width) × \(picture.height) × \(factor) → \(size.width) × \(size.height)"
              + (sizeOK ? "" : ", expected \(expected)"))
        ok = sizeOK && ok
        let input = picture.input(width: size.width, height: size.height, softness: 0)
        ok = Verify.report("picture input", got: input.pixels, want: try reference("input").transposed(0, 2, 3, 1)) && ok
        let softness = (json["softness"] as? NSNumber)?.doubleValue ?? 0.5
        let softened = picture.input(width: size.width, height: size.height, softness: softness)
        ok = Verify.report("softened input", got: softened.pixels, want: try reference("softened").transposed(0, 2, 3, 1)) && ok

        // 2. The encode, in tiles, from the reference's input.
        let wantedLatent = try reference("latent")[0..., 0..., 0].transposed(0, 2, 3, 1)
        let latent = try model.encodeTiled(try reference("input").transposed(0, 2, 3, 1))
        eval(latent)
        ok = Verify.report("vae encode (tiles)", got: latent, want: wantedLatent) && ok

        // 3. The noise and the transformer's input.
        let seed = (json["seed"] as? NSNumber)?.intValue ?? 0
        let noise = SeedVR2Model.noise(seed: seed, latentHeight: latent.shape[1], latentWidth: latent.shape[2], channels: config.latentChannels)
        ok = Verify.report("initial noise", got: noise, want: try reference("noise")) && ok
        let modelInput = SeedVR2Model.modelInput(noise: try reference("noise"), latent: wantedLatent)
        ok = Verify.report("model input", got: modelInput, want: try reference("model_input")) && ok

        // 4. One pass and the step.
        let flow = model.transformer(vid: try reference("model_input"), txt: try reference("text"), timestep: 1000)
        eval(flow)
        ok = Verify.report("transformer pass", got: flow, want: try reference("flow")) && ok
        let latents = SeedVR2Model.step(latents: try reference("noise"), flow: try reference("flow"), index: 0, steps: 1)
        ok = Verify.report("step", got: latents, want: try reference("latents")) && ok

        // 5. The decode, in tiles, cropped to the picture.
        let (width, height) = (size.width, size.height)
        let grid = try reference("latents")[0..., 0..., 0].transposed(0, 2, 3, 1)
        let decoded = try VAETiling.decode(grid, decode: { model.vae.decode($0) })[0..., 0 ..< height, 0 ..< width, 0...]
        eval(decoded)
        let wantedDecoded = try reference("decoded").transposed(0, 2, 3, 1)
        ok = Verify.report("vae decode (tiles)", got: decoded, want: wantedDecoded) && ok

        // 6. The color correction on the reference's decode, then the bytes.
        let style = try reference("input").transposed(0, 2, 3, 1)[0..., 0 ..< height, 0 ..< width, 0...]
        let corrected = SeedVR2ColorCorrection.apply(content: wantedDecoded.asType(.float32).asArray(Float.self),
                                                     style: style.asType(.float32).asArray(Float.self), width: width, height: height)
        let correctedArray = MLXArray(corrected, [1, height, width, 3])
        ok = Verify.report("color correction", got: correctedArray, want: try reference("corrected").transposed(0, 2, 3, 1)) && ok
        ok = Verify.report("pixels", got: Pixels.toPixels(correctedArray), want: try reference("pixels")) && ok

        // 7. The whole upscale from the picture, as the app runs it. The encode's and decode's
        // bfloat16 rounding goes through the random weights and the histogram match, so a few
        // pixels move: shown, with the PSNR, but the stages above decide.
        let image = try model.upscale(picture: picture, width: width, height: height, softness: 0, seed: seed,
                                      phase: { _ in }, progress: { _, _ in }, isCancelled: { false })
        _ = Verify.report("whole upscale", got: image.pixels, want: try reference("pixels"), counts: false)
        let mse = square(image.pixels.asType(.float32) - (try reference("pixels")).asType(.float32)).mean().item(Float.self)
        print(String(format: "    whole upscale PSNR %.1f dB", 10 * log10(255 * 255 / Swift.max(mse, 1e-10))))
        return ok
    }
}

extension SeedVR2Config {
    /// The fixture's small geometry (make_seedvr2_fixture.py's CONFIG).
    static func fixture(_ json: [String: Any]) -> SeedVR2Config {
        var config = SeedVR2Config()
        config.name = "seedvr2 fixture"
        let t = json["transformer"] as? [String: Any] ?? [:]
        func int(_ key: String, _ fallback: Int) -> Int { (t[key] as? NSNumber)?.intValue ?? fallback }
        config.vidDim = int("vid_dim", config.vidDim)
        config.txtInDim = int("txt_in_dim", config.txtInDim)
        config.heads = int("heads", config.heads)
        config.headDim = int("head_dim", config.headDim)
        config.numLayers = int("num_layers", config.numLayers)
        config.mmLayers = int("mm_layers", config.mmLayers)
        config.ropeDim = int("rope_dim", config.ropeDim)
        if let channels = (json["vae"] as? [String: Any])?["block_out_channels"] as? [NSNumber] {
            config.blockOutChannels = channels.map(\.intValue)
        }
        return config
    }
}
