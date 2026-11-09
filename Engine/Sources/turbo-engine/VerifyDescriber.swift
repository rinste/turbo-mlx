import Foundation
import MLX
import TurboEngineCore

/// Describe Image's Qwen2.5-VL against mlx-vlm on the pack itself (Engine/Reference/qwen25vl_reference.py), picture by
/// picture: the pixel values the picture is read into, exactly; the prompt's ids, exactly; the vision tower's tokens
/// from the reference's pixels; the first step's logits from the reference's ids, and the token they choose; then the
/// tokens written greedily, counted as long as they are the reference's, and the text. The two round float16 apart (the
/// builds of MLX, and the language model's norms, which are Qwen-Image's: float32 as transformers computes them, where
/// mlx-vlm's fused norm rounds before the weight), so a near tie may choose another word: the written tokens and the
/// text are reported, not judged.
enum VerifyDescriber {
    static func run(fixture: URL, json: [String: Any], references: [String: MLXArray]) throws -> Bool {
        func reference(_ name: String) throws -> MLXArray {
            guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
            return array
        }
        func exact(_ stage: String, _ got: MLXArray, _ want: MLXArray) -> Bool {
            let same = got.shape == want.shape && (got .== want).all().item(Bool.self)
            print("  \(stage): \(same ? "identical" : "DIFFERENT \(got.shape) vs \(want.shape)")")
            return same
        }
        var ok = true
        // With float32 references (truth.safetensors), the vision tokens pass when the port is no further from them than
        // mlx-vlm is, give or take half, and the first logits give or take double: the norms move the rounding by as
        // much either way (0.7 to 1.6 times mlx-vlm's distance on the pictures of 9 November 2026, on the same MLX).
        let truths = (try? loadArrays(url: fixture.appending(path: "truth.safetensors"))) ?? [:]
        func judged(_ stage: String, _ got: MLXArray, _ want: MLXArray, truth name: String, slack: Float = 1.5) -> Bool {
            guard let truth = truths[name] else { return Verify.report(stage, got: got, want: want) }
            _ = Verify.report(stage, got: got, want: want, counts: false)
            let ours = VerifyMusic3.relativeRMS(got, truth), theirs = VerifyMusic3.relativeRMS(want, truth)
            let pass = ours <= slack * theirs + 1e-4
            print(String(format: "      from float32: ours %.5f, mlx-vlm's %.5f%@", ours, theirs, pass ? "" : "  ✗ FURTHER"))
            return pass
        }
        let instruction = json["instruction"] as? String ?? Qwen25VLDescriber.instruction
        if instruction != Qwen25VLDescriber.instruction { print("  (the fixture's instruction is not the describer's: its own is used)") }
        let minPixels = (json["min_pixels"] as? NSNumber)?.intValue ?? Qwen25VLPicture.minPixels
        let maxPixels = (json["max_pixels"] as? NSNumber)?.intValue ?? Qwen25VLPicture.maxPixels
        let maxTokens = (json["max_tokens"] as? NSNumber)?.intValue ?? Qwen25VLDescriber.maxTokens

        let pack = URL(fileURLWithPath: json["pack"] as? String ?? fixture.path)
        print("loading \(pack.path)…")
        let started = Date()
        let describer = try Qwen25VLDescriber(folder: pack)
        print(String(format: "  (loaded in %.1f s)", Date().timeIntervalSince(started)))

        // Decoding gives back what was encoded, characters cut into bytes across tokens included.
        let texts = ["A sign that reads \"Le Petit Four\".\n\nThe café's menu: crème brûlée, 東京の夜景, 🎉🧁, ąę, Ελληνικά, עברית, ∑ ≤ ∞",
                     "  spaces,\ttabs and\r\nnew lines  ", "<|im_start|>a special token typed by hand"]
        let roundTrips = texts.filter { describer.decode(describer.tokenIds(text: $0)) == $0 }.count
        print("  decoding: \(roundTrips) of \(texts.count) texts round-trip")
        ok = roundTrips == texts.count && ok
        var agreeing = 0, written = 0, sameTexts = 0
        let pictures = json["pictures"] as? [[String: Any]] ?? []
        for entry in pictures {
            guard let name = entry["name"] as? String, let path = entry["path"] as? String else { continue }
            print("\(name):")
            Memory.peakMemory = 0

            // The picture, read and prepared here, against the reference's pixels.
            let picture = try Qwen25VLPicture(path: path)
            let (pixelValues, grid) = picture.visionInput(describer.config.vision, minPixels: minPixels, maxPixels: maxPixels)
            let refGrid = try reference("\(name).grid").reshaped([-1]).asArray(Int32.self).map(Int.init)
            let sameGrid = [grid.t, grid.h, grid.w] == refGrid
            print("  grid \(grid.t) × \(grid.h) × \(grid.w): \(sameGrid ? "the reference's" : "DIFFERENT from \(refGrid)")")
            ok = sameGrid && ok
            let refPixels = try reference("\(name).pixel_values")
            ok = exact("pixel values", pixelValues, refPixels) && ok
            if pixelValues.shape == refPixels.shape { _ = Verify.report("pixel values", got: pixelValues, want: refPixels, counts: false) }

            // The prompt.
            let refIDs = try reference("\(name).ids").reshaped([-1])
            let pictureTokens = grid.patches / (describer.config.vision.spatialMergeSize * describer.config.vision.spatialMergeSize)
            let ids = describer.tokenIds(pictureTokens: pictureTokens, instruction: instruction)
            ok = exact("prompt ids (\(ids.count))", MLXArray(ids.map { Int32($0) }), refIDs.asType(.int32)) && ok

            // The vision tower and the first step, from the reference's inputs.
            let refGridValue = QwenVisionGrid(t: refGrid[0], h: refGrid[1], w: refGrid[2])
            let embeds = describer.pictureEmbeds(pixelValues: refPixels, grid: refGridValue)
            eval(embeds)
            ok = judged("vision tokens", embeds, try reference("\(name).vision"), truth: "\(name).vision") && ok
            let refTokens = try reference("\(name).tokens").asArray(Int32.self).map(Int.init)
            var logits: MLXArray?
            let writing = Date()
            let tokens = try describer.generate(
                ids: refIDs.asArray(Int32.self).map(Int.init), pictureEmbeds: embeds, grid: refGridValue, maxTokens: maxTokens,
                firstLogits: { logits = $0 }
            )
            let seconds = Date().timeIntervalSince(writing)
            let refLogits = try reference("\(name).logits")
            if let logits {
                ok = judged("first logits", logits, refLogits, truth: "\(name).logits", slack: 2) && ok
                let ours = argMax(logits, axis: 0).item(Int.self), theirs = argMax(refLogits, axis: 0).item(Int.self)
                print("  first token: \(ours == theirs ? "the reference's" : "DIFFERENT (\(ours) against \(theirs))")")
                ok = ours == theirs && ok
            }

            // The answer: the same tokens as long as no near tie goes the other way.
            let common = zip(tokens, refTokens).prefix { $0 == $1 }.count
            agreeing += common
            written += refTokens.count
            let text = describer.decode(tokens).trimmingCharacters(in: .whitespacesAndNewlines)
            let sameText = text == (entry["text"] as? String ?? "")
            if sameText { sameTexts += 1 }
            if common == tokens.count, common == refTokens.count {
                print("  written: all \(common) tokens the reference's")
            } else {
                // How close the two candidates were where they part: a near tie, or a real difference.
                var gap = ""
                if common < tokens.count, common < refTokens.count {
                    var at: MLXArray?
                    _ = try describer.generate(ids: refIDs.asArray(Int32.self).map(Int.init) + Array(refTokens[0 ..< common]),
                                               pictureEmbeds: embeds, grid: refGridValue, maxTokens: 0, firstLogits: { at = $0 })
                    if let at {
                        let values = at.asArray(Float.self)
                        gap = String(format: " (their logits there: %.3f for ours, %.3f for the reference's)",
                                     values[tokens[common]], values[refTokens[common]])
                    }
                }
                print("  written: the first \(common) of \(refTokens.count) tokens the reference's, \(tokens.count) in all\(gap)")
            }
            print(String(format: "  (%d tokens in %.2f s, peak %.2f GB)", tokens.count, seconds, Double(Memory.peakMemory) / 1e9))
            print("  text: \(sameText ? "identical" : "different")\n    \(text)")
            Memory.clearCache()
        }
        print("tokens written as the reference's: \(agreeing) of \(written); texts identical: \(sameTexts) of \(pictures.count)")
        return ok
    }
}
