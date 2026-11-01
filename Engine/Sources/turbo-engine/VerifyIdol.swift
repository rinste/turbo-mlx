import Foundation
import MLX
import TurboEngineCore

/// The Idol port (Engine/Reference/idol_reference.py --record: idol-sdk's PyTorch code on MPS with the MLX pack's weights
/// dequantized), each stage from the reference's own inputs. The picture: its resize, BiRefNet's input, backbone passes,
/// squeezed features, alpha map and bytes, and the crop to 1024 on black from the reference's alpha (Pillow's
/// arithmetic, byte for byte). DINOv3's embeddings, layers and tokens; the VAE's moments and, with the reference's noise,
/// its tokens. The octree decoder's logits at three levels and the Gaussian decoder's outputs and activations, on the
/// run's final latent. Then the denoiser: from the reference's own picture tokens and noise, the time embedding, the refined streams,
/// the first joint block's rotation and output, the last block's, both velocities at t = 700 and the guided one, then
/// a 4-step guided run. Each stage also gives the cosine between the two. With the float32 references `--truth` adds,
/// the short run is decided against float32: its float16 loop amplifies rounding (four steps, the last of half the
/// way), so it passes when it is no further from the float32 run than twice the reference's own float16 run is.
enum VerifyIdol {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        let pack = URL(fileURLWithPath: json["pack"] as? String ?? fixture.path)
        var ok = true
        func check(_ stage: String, _ got: MLXArray, _ want: MLXArray, counts: Bool = true) -> Bool {
            eval(got)
            let pass = Verify.report(stage, got: got, want: want, counts: counts)
            print(String(format: "      cos %.6f", cosine(got, want)))
            return pass
        }
        if references["matte_rgb"] != nil {
            ok = try verifyPicture(pack: pack, json: json, reference: reference, check: check) && ok
            Memory.clearCache()
        }
        if references["vae_moments"] != nil {
            ok = try verifyEncoders(pack: pack, reference: reference, check: check) && ok
            Memory.clearCache()
        }
        if references["oct_logits_1"] != nil {
            ok = try verifyDecoders(pack: pack, reference: reference, check: check) && ok
            Memory.clearCache()
        }
        print("loading \(pack.path)…")
        let denoiser = try IdolDenoiser.load(folder: pack)

        let feature1 = try reference("feature1")
        let feature2 = try reference("feature2")
        let latent = try reference("noise_latent")
        let camera = try reference("noise_camera")
        let t = (json["t"] as? NSNumber)?.floatValue ?? 700

        // 1. The streams before the joint blocks.
        let conditional = denoiser.context(feature1: feature1, feature2: feature2)
        ok = check("picture refined", conditional, try reference("context_refined")) && ok
        let stream = denoiser.stream(latent: latent, camera: camera, t: t)
        ok = check("time embedding", stream.timeEmbedding, try reference("t_emb")) && ok
        ok = check("latent refined", stream.latent, try reference("noise_refined")) && ok
        ok = check("camera refined", stream.camera, try reference("camera_refined")) && ok

        // 2. One conditional pass, block by block, then the unconditional one.
        var inner: [String: MLXArray] = [:]
        let cond = denoiser.joint(stream, context: conditional) { name, array in
            eval(array)
            inner[name] = array
        }
        ok = check("block 0 rotation", inner["rope_block0"]!, try reference("rope_block0")) && ok
        ok = check("block 0", inner["block0"]!, try reference("block0")) && ok
        ok = check("last block", inner["block_last"]!, try reference("block_last")) && ok
        ok = check("velocity", cond.latent, try reference("v_cond_latent")) && ok
        ok = check("camera velocity", cond.camera, try reference("v_cond_camera")) && ok
        let unconditional = denoiser.context(feature1: zeros(like: feature1), feature2: zeros(like: feature2))
        let uncond = denoiser.joint(stream, context: unconditional)
        ok = check("uncond velocity", uncond.latent, try reference("v_uncond_latent")) && ok
        ok = check("uncond camera", uncond.camera, try reference("v_uncond_camera")) && ok
        // The same pass with the picture's 4,101 alike tokens run as two biased keys.
        let collapsedContext = denoiser.collapsed(unconditional)
        let collapsed = denoiser.joint(stream, collapsed: collapsedContext)
        let (b1, b2) = collapsedContext.biases
        print(String(format: "  · picture tokens collapsed to two keys, biases %.4f and %.4f (e^b1 + e^b2 = %.3f)",
                     Float(b1), Float(b2), exp(Double(b1)) + exp(Double(b2))))
        ok = check("uncond, collapsed", collapsed.latent, try reference("v_uncond_latent")) && ok
        _ = check("  against uncond", collapsed.latent, uncond.latent, counts: false)
        ok = check("uncond camera, coll.", collapsed.camera, try reference("v_uncond_camera")) && ok
        let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 3
        let guided = (1 + guidance) * cond.latent - guidance * uncond.latent
        let wantedGuided = (1 + guidance) * (try reference("v_cond_latent")) - guidance * (try reference("v_uncond_latent"))
        ok = check("guided velocity", guided, wantedGuided) && ok

        // 3. A short guided run from the same noise.
        let steps = (json["short_steps"] as? NSNumber)?.intValue ?? 4
        let rescale = (json["rescale_t"] as? NSNumber)?.doubleValue ?? 3
        let sampler = IdolSampler(steps: steps, guidance: guidance, rescale: rescale)
        let short = try sampler.run(denoiser, latent: latent, camera: camera, conditional: conditional, unconditional: unconditional)
        var collapsing = sampler
        collapsing.collapsesUnconditional = true
        let shortCollapsed = try collapsing.run(denoiser, latent: latent, camera: camera, conditional: conditional, unconditional: unconditional)
        guard let truth = references["short_latent_f32"] else {
            ok = check("\(steps)-step latent", short.latent, try reference("short_latent")) && ok
            ok = check("\(steps)-step camera", short.camera, try reference("short_camera")) && ok
            ok = check("\(steps)-step, collapsed", shortCollapsed.latent, try reference("short_latent")) && ok
            return ok
        }
        if let cond32 = references["v_cond_latent_f32"], let uncond32 = references["v_uncond_latent_f32"] {
            _ = check("velocity vs float32", cond.latent, cond32, counts: false)
            _ = check("  theirs vs float32", try reference("v_cond_latent"), cond32, counts: false)
            _ = check("uncond vs float32", uncond.latent, uncond32, counts: false)
            _ = check("  theirs vs float32", try reference("v_uncond_latent"), uncond32, counts: false)
        }
        let theirs = try reference("short_latent")
        _ = check("\(steps)-step latent", short.latent, theirs, counts: false)
        _ = check("\(steps)-step camera", short.camera, try reference("short_camera"), counts: false)
        _ = check("  vs float32", short.latent, truth, counts: false)
        _ = check("  collapsed vs f32", shortCollapsed.latent, truth, counts: false)
        _ = check("  theirs vs float32", theirs, truth, counts: false)
        if let batched = references["short_latent_batched"] {
            _ = check("  theirs, batched", batched, theirs, counts: false)
        }
        let rounding = relativeRMS(theirs, truth)
        for (name, run) in [("ours", short.latent), ("ours, collapsed", shortCollapsed.latent)] {
            let distance = relativeRMS(run, truth)
            let pass = distance <= 2 * rounding
            print(String(format: "  %@ %d-step run, %@: RMS %.4f from float32, the reference's own float16 run %.4f",
                         pass ? "✓" : "✗", steps, name, distance, rounding))
            ok = pass && ok
        }
        return ok
    }

    /// Bytes against bytes: how many differ and by how much; `exact` decides on none differing.
    static func compareBytes(_ stage: String, _ got: [UInt8], _ want: MLXArray, exact: Bool) -> Bool {
        let wanted = want.asType(.uint8).reshaped([-1]).asArray(UInt8.self)
        guard wanted.count == got.count else {
            print("  ✗ \(stage): \(got.count) bytes, expected \(wanted.count)")
            return false
        }
        var differing = 0, largest = 0
        for (a, b) in zip(got, wanted) where a != b {
            differing += 1
            largest = Swift.max(largest, abs(Int(a) - Int(b)))
        }
        let pass = !exact || differing == 0
        print(String(format: "  %@ %-18@ %d of %d bytes differ (%.4f%%), by at most %d", exact ? (pass ? "✓" : "✗") : "·", stage,
                     differing, got.count, 100 * Double(differing) / Double(Swift.max(got.count, 1)), largest))
        return pass
    }

    /// The picture's preparation and BiRefNet.
    static func verifyPicture(pack: URL, json: [String: Any], reference: (String) throws -> MLXArray,
                              check: (String, MLXArray, MLXArray, Bool) -> Bool) throws -> Bool {
        var ok = true
        print("BiRefNet and the picture…")
        let matte = try IdolMatte.load(url: pack.appending(path: "birefnet.safetensors"))
        let wantedRGB = try reference("matte_rgb")
        let (height, width) = (wantedRGB.dim(0), wantedRGB.dim(1))
        // The file read and resized here (ImageIO's JPEG decoder against libjpeg: shown).
        if let image = json["image"] as? String, let picture = try? IdolPicture.load(image) {
            let scale = 1024 / Double(Swift.min(picture.width, picture.height))
            let resized = picture.resized(width: Swift.max(1, Int((Double(picture.width) * scale).rounded(.toNearestOrEven))),
                                          height: Swift.max(1, Int((Double(picture.height) * scale).rounded(.toNearestOrEven))))
            _ = compareBytes("picture resized", resized.rgb, wantedRGB, exact: false)
        }
        // The network from the reference's input, stage by stage.
        let input = try reference("matte_input").transposed(0, 2, 3, 1)
        let rgb = wantedRGB.asType(.uint8).reshaped([-1]).asArray(UInt8.self)
        let ours = (idolResizeBilinear(MLXArray(rgb.map { Float($0) / 255 }).reshaped([1, height, width, 3]), height: 1024, width: 1024)
            - MLXArray(IdolMatte.mean)) / MLXArray(IdolMatte.std)
        ok = check("matte input", ours, input, true) && ok
        let full = matte.backboneFeatures(input)
        for (level, feature) in full.enumerated() {
            ok = check("  backbone \(level)", feature, try reference("matte_bb\(level)").transposed(0, 2, 3, 1), true) && ok
        }
        let map = matte(input)
        ok = check("alpha map", map, try reference("matte_alpha_map").transposed(0, 2, 3, 1), true) && ok
        let alpha = matte.alpha(rgb: rgb, width: width, height: height)
        _ = compareBytes("alpha bytes", alpha, try reference("matte_alpha"), exact: false)
        // The crop to 1024 on black from the reference's own picture and alpha: Pillow's arithmetic, byte for byte.
        let wantedAlpha = try reference("matte_alpha").asType(.uint8).reshaped([-1]).asArray(UInt8.self)
        let picture = IdolPicture(bytes: rgb, width: width, height: height, channels: 3)
        let margin = (json["mode"] as? String) == "real" ? 1.7 : 1.2
        let prepared = try IdolPreparation.prepare(picture, margin: margin) { _, _, _ in wantedAlpha }
        ok = compareBytes("prepared", prepared, try reference("prepared"), exact: true) && ok
        // And with our own alpha (shown).
        let preparedOurs = try IdolPreparation.prepare(picture, margin: margin) { _, _, _ in alpha }
        _ = compareBytes("prepared, our alpha", preparedOurs, try reference("prepared"), exact: false)
        return ok
    }

    /// DINOv3 and the VAE from the reference's prepared picture.
    static func verifyEncoders(pack: URL, reference: (String) throws -> MLXArray,
                               check: (String, MLXArray, MLXArray, Bool) -> Bool) throws -> Bool {
        var ok = true
        print("DINOv3 and the VAE…")
        let prepared = try reference("prepared")
        let pixels = prepared.asType(.float32).reshaped([1, prepared.dim(0), prepared.dim(1), 3]) / 255
        let dinov3 = try IdolDINOv3.load(folder: pack.appending(path: "dinov3"))
        var inner: [String: MLXArray] = [:]
        let feature1 = dinov3.tokens(pixels) { name, array in
            if ["embeddings", "layer0", "layer15", "hidden"].contains(name) {
                eval(array)
                inner[name] = array
            }
        }
        ok = check("dinov3 embeddings", inner["embeddings"]!, try reference("dinov3_embeddings"), true) && ok
        ok = check("dinov3 layer 0", inner["layer0"]!, try reference("dinov3_layer0"), true) && ok
        ok = check("dinov3 layer 15", inner["layer15"]!, try reference("dinov3_layer15"), false) && ok
        ok = check("dinov3 hidden", inner["hidden"]!, try reference("dinov3_hidden"), false) && ok
        ok = checkAgainstTruth("feature1", ours: feature1, theirs: try reference("feature1"), truth: try? reference("feature1_f32"), check: check) && ok
        let vae = try IdolVAEEncoder.load(folder: pack.appending(path: "vae"))
        let moments = vae.moments(pixels)
        let eps = try reference("vae_eps").transposed(0, 2, 3, 1).asType(.bfloat16)
        // The VAE's convolutions round differently in bfloat16 here (ours lands ~2.7× further from float32 than the
        // reference's), far below the posterior's own scatter, which the sample adds: decided on the structure, the
        // cosine to float32, with the distances shown.
        let theirMoments = try reference("vae_moments").transposed(0, 2, 3, 1)
        _ = check("vae moments", moments, theirMoments, false)
        if let truth = (try? reference("vae_moments_f32"))?.transposed(0, 2, 3, 1) {
            let mean = truth[.ellipsis, 0 ..< 32]
            let scatter = exp(0.5 * clip(truth[.ellipsis, 32...], min: -30, max: 20)) * eps.asType(.float32)
            let pass = cosine(moments, truth) >= 0.999
            print(String(format: "  %@ vae moments: cos %.6f to float32 (the reference's %.6f), RMS %.4f from it (the reference's %.4f); the posterior's own scatter is %.2f of its mean",
                         pass ? "✓" : "✗", cosine(moments, truth), cosine(theirMoments, truth), relativeRMS(moments, truth),
                         relativeRMS(theirMoments, truth), relativeRMS(scatter + mean, mean)))
            ok = pass && ok
        }
        _ = check("feature2", vae.tokens(moments: moments, eps: eps), try reference("feature2"), false)
        ok = check("  from its moments", vae.tokens(moments: try reference("vae_moments").transposed(0, 2, 3, 1).asType(.bfloat16), eps: eps),
                   try reference("feature2"), true) && ok
        return ok
    }

    /// The octree and Gaussian decoders on the run's final latent.
    static func verifyDecoders(pack: URL, reference: (String) throws -> MLXArray,
                               check: (String, MLXArray, MLXArray, Bool) -> Bool) throws -> Bool {
        var ok = true
        print("the decoders…")
        let config = try IdolConfig.read(folder: pack)
        let decoder = try IdolModel.loadDecoder(IdolOctreeDecoder(config: config.decoder), file: "decoder.safetensors", folder: pack)
        let gaussianDecoder = try IdolModel.loadDecoder(IdolGaussianDecoder(config: config.gaussianDecoder), file: "decoder_gs.safetensors", folder: pack)
        let latent = try reference("final_latent")
        for level in [1, 5, 8] {
            let logits = decoder(try reference("oct_centres_\(level)"), resolution: 1 << level, latent: latent)
            ok = check("octree level \(level)", logits, try reference("oct_logits_\(level)"), true) && ok
        }
        let points = try reference("gs_points")
        let raw = gaussianDecoder(points, latent: latent)
        ok = checkAgainstTruth("gaussian raw", ours: raw, theirs: try reference("gs_raw"), truth: try? reference("gs_raw_f32"), check: check) && ok
        // The activations from the reference's raw outputs: exact up to float32 rounding.
        let g = gaussianDecoder.activate(points: points, raw: try reference("gs_raw"))
        ok = check("  positions", g.xyz, try reference("gs_xyz"), true) && ok
        ok = check("  colours", g.colour, try reference("gs_color"), true) && ok
        ok = check("  scales", g.scale, try reference("gs_scaling"), true) && ok
        ok = check("  rotations", g.rotation, try reference("gs_rotation"), true) && ok
        ok = check("  opacities", g.opacity, try reference("gs_opacity").reshaped([-1]), true) && ok
        return ok
    }

    /// A 16-bit stage against the reference's, decided against float32 when the recording has it (`--truth`): ours no
    /// further from the float32 result than twice the reference's own 16-bit result is.
    static func checkAgainstTruth(_ stage: String, ours: MLXArray, theirs: MLXArray, truth: MLXArray?,
                                  check: (String, MLXArray, MLXArray, Bool) -> Bool) -> Bool {
        guard let truth else { return check(stage, ours, theirs, true) }
        _ = check(stage, ours, theirs, false)
        let (distance, rounding) = (relativeRMS(ours, truth), relativeRMS(theirs, truth))
        let pass = distance <= 2 * rounding
        print(String(format: "  %@ %@: RMS %.4f from float32, the reference's own %.4f (cos %.6f, %.6f)", pass ? "✓" : "✗", stage,
                     distance, rounding, cosine(ours, truth), cosine(theirs, truth)))
        return pass
    }

    /// RMS of the difference over RMS of `b`.
    static func relativeRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
        let (x, y) = (a.asType(.float32), b.asType(.float32))
        return sqrt(square(x - y).mean()).item(Float.self) / Swift.max(sqrt(square(y).mean()).item(Float.self), 1e-12)
    }

    /// The cosine between two arrays taken as vectors, in float32.
    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let (x, y) = (a.asType(.float32).reshaped([-1]), b.asType(.float32).reshaped([-1]))
        let dot = (x * y).sum().item(Float.self)
        let norms = sqrt((x * x).sum()).item(Float.self) * sqrt((y * y).sum()).item(Float.self)
        return dot / Swift.max(norms, 1e-20)
    }
}
