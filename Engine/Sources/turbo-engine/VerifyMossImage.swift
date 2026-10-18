import Foundation
import MLX
import TurboEngineCore

/// The MOSS-Image port (Engine/Fixtures/make_moss_image_fixture.py, OpenMOSS's `pipeline_moss_image.py`
/// in float32): the Qwen3.5 encoder layer by layer for prompts shorter and longer than the delta
/// rule's chunk, the snapped schedule for several step counts, the DiT's rotary table and one forward
/// (plain, long prompt, block 22 skipped), the loop with guidance and without, then the decode.
enum VerifyMossImage {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        if json["pack"] != nil { return try runRecording(json: json, reference: reference) }
        print("loading \(fixture.path)…")
        let model = try MossImageModel(modelPath: fixture, loadTokenizer: false)
        let config = model.config
        var ok = true

        // 1. The text encoder, from the reference's token ids.
        let encoder = try model.textEncoder()
        let promptCount = (json["prompt_lengths"] as? [Any])?.count ?? 1
        for tag in (0 ..< promptCount).map({ "prompt\($0)" }) + ["negative"] {
            let ids = try reference("\(tag)_input_ids").asType(.int32)
            var layers: [Int: MLXArray] = [:]
            let hidden = encoder.lastHiddenState(ids, computeType: .float32) { layers[$0] = $1 }
            for (index, state) in layers.sorted(by: { $0.key < $1.key }) {
                let kind = config.textEncoder.fullAttention[index] ? "attention" : "delta net"
                _ = Verify.report("  \(tag) layer \(index) (\(kind))", got: state, want: try reference("\(tag)_layer_\(index)"), counts: false)
            }
            ok = Verify.report("\(tag) hidden (\(ids.dim(1)) tokens)", got: hidden, want: try reference("\(tag)_hidden")) && ok
        }

        // 2. The schedule, which must match to the bit: it picks points of the training grid.
        for steps in (json["schedule_steps"] as? [NSNumber] ?? []).map(\.intValue) {
            let schedule = MossImageSchedule(steps: steps, config: config)
            func same(_ got: [Float], _ name: String) throws -> Bool {
                let wanted = try reference(name)
                return wanted.shape == [got.count] && (MLXArray(got) .== wanted).all().item(Bool.self)
            }
            let exact = try same(schedule.sigmas, "sigmas_\(steps)")
            let timesExact = try same(schedule.timesteps, "timesteps_\(steps)")
            print("  \(exact && timesExact ? "✓" : "✗") schedule \(steps) steps: \(schedule.steps) sigmas\(exact && timesExact ? ", bit-exact" : " differ")")
            ok = exact && timesExact && ok
        }

        // 3. The rotary table and one forward, on the reference's own conditioning.
        let noise = try reference("noise").transposed(0, 2, 3, 1)
        let (height, width) = (noise.dim(1), noise.dim(2))
        let prompt = try reference("prompt0_hidden")
        let negative = try reference("negative_hidden")
        let rope = try model.transformer.rope(height: height, width: width, textLength: prompt.dim(1), dtype: .float32)
        ok = Verify.report("rope", got: rope, want: try reference("rope")) && ok
        let timestep = (json["timestep"] as? NSNumber)?.floatValue ?? 873
        func channelsFirst(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }
        ok = Verify.report("forward", got: channelsFirst(try model.transformer(noise, timestep: timestep, context: prompt)),
                           want: try reference("forward")) && ok
        ok = Verify.report("forward, long prompt", got: channelsFirst(try model.transformer(noise, timestep: timestep, context: try reference("prompt1_hidden"))),
                           want: try reference("forward_long")) && ok
        ok = Verify.report("forward, skip 22", got: channelsFirst(try model.transformer(noise, timestep: timestep, context: negative, skipLayers: [22])),
                           want: try reference("forward_skip")) && ok

        // 4. The loop from the reference's noise, with guidance and without.
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 4
        let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 3
        let guided = try model.denoise(noise: noise, prompt: prompt, negative: negative, steps: steps, guidance: guidance)
        ok = Verify.report("latents", got: channelsFirst(guided), want: try reference("latents")) && ok
        let unguided = try model.denoise(noise: noise, prompt: prompt, negative: nil, steps: steps, guidance: 1)
        ok = Verify.report("latents, guidance 1", got: channelsFirst(unguided), want: try reference("latents_no_guidance")) && ok

        // 5. The decode of the reference's latents (bf16 here, float32 there), in 0…255.
        let pixels = model.decode(try reference("latents").transposed(0, 2, 3, 1)).asType(.float32)
        let wantedPixels = (try reference("image")[0].transposed(1, 2, 0) * 255).round()
        // A random decoder turns bf16 rounding into a few percent of the image (the reference's own
        // bf16 decode is that far from its float32 one), so the decode passes when it is no further
        // from the float32 image than twice what rounding moves the reference itself.
        if let bf16 = references["image_bf16"] {
            let rounded = (bf16[0].transposed(1, 2, 0) * 255).round()
            _ = Verify.report("image", got: pixels, want: wantedPixels, counts: false)
            _ = Verify.report("  reference in bf16", got: rounded, want: wantedPixels, counts: false)
            func rms(_ a: MLXArray, _ b: MLXArray) -> Float { sqrt(square(a - b).mean()).item(Float.self) }
            let (port, rounding) = (rms(pixels, wantedPixels), rms(rounded, wantedPixels))
            let pass = port <= 2 * rounding + 0.5
            print(String(format: "  %@ image RMS %.2f of 255, the reference's own bf16 decode %.2f", pass ? "✓" : "✗", port, rounding))
            ok = pass && ok
        } else {
            ok = Verify.report("image", got: pixels, want: wantedPixels) && ok
        }
        return ok
    }

    /// A recording of OpenMOSS's pipeline on a real checkpoint (Engine/Reference/moss_image_reference.py
    /// --record): the tokens must be the same; the rest is compared for information, in bf16 (or the
    /// bits `quantize` in fixture.json names) where rounding alone moves 32 + 42 layers apart. The
    /// loops start from the reference's noise and its own prompt states, then from the port's.
    static func runRecording(json: [String: Any], reference: (String) throws -> MLXArray) throws -> Bool {
        let pack = URL(fileURLWithPath: json["pack"] as? String ?? "")
        let quantize = (json["quantize"] as? NSNumber)?.intValue
        print("loading \(pack.path)\(quantize.map { ", quantized to \($0) bits" } ?? "")…")
        let model = try MossImageModel(modelPath: pack, quantize: quantize)
        var ok = true

        let prompt = json["prompt"] as? String ?? ""
        let negativeText = json["negative"] as? String ?? ""
        var hidden: [String: MLXArray] = [:]
        for (tag, text) in [("prompt", prompt), ("negative", negativeText)] {
            let ids = try model.tokenIds(text)
            let wanted = try reference("\(tag)_input_ids").reshaped([-1]).asArray(Int32.self).map(Int.init)
            let same = ids == wanted
            print("  \(same ? "✓" : "✗") \(tag) tokens: \(ids.count)\(same ? ", identical" : " against \(wanted.count)")")
            ok = same && ok
            hidden[tag] = try model.encodePrompt(text)
            _ = Verify.report("\(tag) hidden", got: hidden[tag]!, want: try reference("\(tag)_hidden"), counts: false)
            // Both bf16 encoders against Qwen3.5 in float32 (moss_image_reference.py --truth).
            if let truth = try? reference("\(tag)_hidden_f32") {
                _ = Verify.report("  ours vs float32", got: hidden[tag]!, want: truth, counts: false)
                _ = Verify.report("  theirs vs float32", got: try reference("\(tag)_hidden"), want: truth, counts: false)
            }
        }

        let noise = try reference("noise").transposed(0, 2, 3, 1)
        let steps = (json["steps"] as? NSNumber)?.intValue ?? 50
        let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 3
        let schedule = MossImageSchedule(steps: steps, config: model.config)
        let dtype = try model.transformer.dtype
        let wantedPrompt = try reference("prompt_hidden").asType(dtype)
        let wantedNegative = try reference("negative_hidden").asType(dtype)
        func channelsFirst(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }
        let conditional = try model.transformer(noise, timestep: schedule.timesteps[0], context: wantedPrompt)
        _ = Verify.report("step 0 velocity", got: channelsFirst(conditional), want: try reference("step0_cond"), counts: false)
        let unconditional = try model.transformer(noise, timestep: schedule.timesteps[0], context: wantedNegative,
                                              skipLayers: model.config.unconditionalSkipLayers)
        _ = Verify.report("step 0 uncond", got: channelsFirst(unconditional), want: try reference("step0_uncond"), counts: false)

        let wantedImage = try reference("image")
        func psnr(_ pixels: MLXArray) -> Float {
            let mse = square(pixels.asType(.float32) - wantedImage).mean().item(Float.self)
            return 10 * log10(255 * 255 / Swift.max(mse, 1e-10))
        }
        let fromReference = try model.denoise(noise: noise, prompt: wantedPrompt, negative: wantedNegative, steps: steps, guidance: guidance)
        _ = Verify.report("latents, its states", got: channelsFirst(fromReference), want: try reference("latents"), counts: false)
        let own = try model.denoise(noise: noise, prompt: hidden["prompt"]!, negative: hidden["negative"]!, steps: steps, guidance: guidance)
        _ = Verify.report("latents", got: channelsFirst(own), want: try reference("latents"), counts: false)
        let decodedReference = model.decode(try reference("latents").transposed(0, 2, 3, 1))
        let fromStates = model.decode(fromReference)
        let image = model.decode(own)
        print(String(format: "  · PSNR vs the reference's image: %.1f dB (its latents, our decode), %.1f dB (its prompt states), %.1f dB (all ours)",
                     psnr(decodedReference), psnr(fromStates), psnr(image)))
        if let out = json["output"] as? String {
            try ImageOutput.writePNG(image, to: URL(fileURLWithPath: out), source: .trainedAlgorithmicMedia, metadata: [:])
        }
        return ok
    }
}
