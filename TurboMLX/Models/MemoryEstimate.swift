import Foundation

/// The memory a generation needs at its peak, so that one this Mac cannot hold is refused before it
/// starts: when the GPU runs out, a Metal allocation failure ends the engine, and the job with it.
/// The figures are MLX's peaks in the engine (mlx-swift 0.32.2), measured on an M1 Max for each
/// catalog model at 512 to 2048 pixels (clips: up to 10 s), with and without Save memory; MLX
/// allocates the same on every Mac. They were measured before the engine wired its memory during a
/// job, which lowers MLX's peak a little (Klein at 1024 px: 10.1 GB against 12.7), so they err on
/// the side of letting a generation start. A model added by hand is not judged.
nonisolated enum MemoryEstimate {
    /// GB at the peak for `megapixels` of output: the larger of what encoding a new prompt takes,
    /// which does not grow with the image, and a line in the image's megapixels.
    private struct Peak {
        var floor: Double
        var base: Double
        var perMegapixel: Double

        func gigabytes(_ megapixels: Double) -> Double { max(floor, base + perMegapixel * megapixels) }
    }

    /// This Mac's memory, in GB.
    static let installedGigabytes = Double(ProcessInfo.processInfo.physicalMemory) / Double(1 << 30)

    /// Gigabytes the engine should peak at for this request on a Mac with `memory` GB, nil when the
    /// model is not measured.
    static func peak(model: ModelDescriptor, size: PixelSize, frames: Int?, lowMemory: Bool, memory: Double = installedGigabytes) -> Double? {
        if model.family == .ltx2, model.isBuiltIn {
            return clipPeak(model: model, size: size, frames: frames ?? 121, lowMemory: lowMemory, memory: memory)
        }
        return peaks(model.id, lowMemory: lowMemory)?.gigabytes(size.megapixels)
    }

    /// What a generation may take on a Mac with `memory` GB: all of it but what macOS and the app keep.
    static func available(memory: Double = installedGigabytes) -> Double { memory - 2.5 }

    /// The Mac's memory in whole gigabytes, as the refusal names it.
    static var installed: Int { Int(installedGigabytes.rounded()) }

    // Measured (29 September 2026): Klein 6.3 / 12.7 / 39.3 GB at 512 / 1024 / 2048 px with or
    // without Save memory (its decoder cannot be tiled); Z-Image 4-bit 7.4 / 12.1 / 31.9 GB, with
    // Save memory 7.4 / 7.7 / 8.7; Qwen-Image 29.1 / 33.3 / 40.0 GB at 512 / 1024 / 1536, with Save
    // memory 16.0 / 16.6 / 17.0 up to 2048; Qwen-Image Edit 30.2 / 32.4, with Save memory 16.3 /
    // 17.2 / 19.9 up to 1536; Ming-Image 21.0 / 25.5, with Save memory 11.7 / 10.4 / 10.5 (the
    // prompt's encoding is the peak); SeedVR2 10.7 GB up to about 5 MP out, 13.2 at 9.4, 18.1 at
    // 16.8. SenseNova-U1.5 4-bit (30 September) 10.2 / 10.8 / 11.2, with Save memory 6.4 / 6.1
    // at 1024 / 2048 once a prompt has been read that way (its two stacks are then never both in
    // memory; the first image after one without Save memory still peaks as without); an edit, whose
    // picture is read with the prompt at the image's size, 10.9 / 11.0 / 11.9, with Save memory
    // 7.4–7.7 / 8.2, which the lines follow. Where a size was not measured without Save memory, the decoder's growth is Qwen-Image's,
    // the same decoder.
    private static func peaks(_ id: String, lowMemory: Bool) -> Peak? {
        switch (id, lowMemory) {
        case ("mflux-community/flux2-klein-4b-mflux-q4", _): Peak(floor: 6.3, base: 3.8, perMegapixel: 8.46)
        case ("mflux-community/z-image-turbo-mflux-q4", false): Peak(floor: 7.4, base: 5.5, perMegapixel: 6.3)
        case ("mflux-community/z-image-turbo-mflux-q4", true): Peak(floor: 7.4, base: 7.37, perMegapixel: 0.32)
        case ("mflux-community/z-image-turbo-mflux-q8", false): Peak(floor: 11.8, base: 10.2, perMegapixel: 6.3)
        case ("mflux-community/z-image-turbo-mflux-q8", true): Peak(floor: 12.3, base: 11.9, perMegapixel: 0.38)
        case ("mflux-community/qwen-image-2512-mflux-q4", false): Peak(floor: 29.1, base: 27.9, perMegapixel: 5.11)
        case ("mflux-community/qwen-image-2512-mflux-q4", true): Peak(floor: 16.0, base: 16.47, perMegapixel: 0.13)
        case ("mflux-community/qwen-image-edit-2511-mflux-q4", false): Peak(floor: 30.2, base: 25.1, perMegapixel: 7.0)
        case ("mflux-community/qwen-image-edit-2511-mflux-q4", true): Peak(floor: 16.3, base: 15.0, perMegapixel: 2.06)
        case ("joeynyc/Ming-Image-0.1-Design-mflux-q8-te5", false): Peak(floor: 21.0, base: 20.15, perMegapixel: 5.11)
        case ("joeynyc/Ming-Image-0.1-Design-mflux-q8-te5", true): Peak(floor: 11.7, base: 10.4, perMegapixel: 0.03)
        case ("numz/SeedVR2_comfyUI", _): Peak(floor: 10.7, base: 7.0, perMegapixel: 0.66)
        case ("mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit", false): Peak(floor: 10.9, base: 10.7, perMegapixel: 0.3)
        case ("mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit", true): Peak(floor: 7.7, base: 7.5, perMegapixel: 0.2)
        default: nil
        }
    }

    // LTX-2.3 (29 September 2026): the 4-bit pack with Save memory peaks at 17.9 / 18.2 / 19.2 GB
    // for 3 / 5 / 10 s at 768 × 512 and at 21.5 GB for 5 s at 1024 × 576; without it at 26.7 / 30.3
    // GB for 1 / 5 s at 768 × 512; the 8-bit pack at 26.8 GB with and 37.4 GB without for 5 s at
    // 768 × 512. A line in the frame's megapixels and in megapixels × latent frames fits them. The
    // decoders get a budget from the engine (`LTXDecodeTiling.budget`: half the memory, within three
    // quarters of it less what is resident), smaller on a smaller Mac, where the clip therefore
    // peaks at most at the transformer's phase or at the decoders plus that budget.
    private static func clipPeak(model: ModelDescriptor, size: PixelSize, frames: Int, lowMemory: Bool, memory: Double) -> Double {
        let heavier = model.id.contains("q8") ? (lowMemory ? 8.6 : 7.1) : 0
        let megapixels = size.megapixels
        let latentFrames = Double((max(frames, 1) - 1) / 8 + 1)
        let fitted = lowMemory
            ? 11.63 + heavier + 14.2 * megapixels + 0.1575 * megapixels * latentFrames
            : 19.9 + heavier + 14.2 * megapixels + 0.763 * megapixels * latentFrames
        // With Save memory the transformer is gone when the decoders run; without it, Gemma and
        // the transformer stay.
        let transformerPhase = (lowMemory ? 17.5 : 22) + heavier
        let residentAtDecode = lowMemory ? 2 : 21 + heavier
        let budget = max(min(memory / 2, memory * 3 / 4 - residentAtDecode), 3)
        return min(fitted, max(transformerPhase, residentAtDecode + budget))
    }

    /// The smallest request the model takes, to tell whether any fits the Mac.
    static func smallestPeak(model: ModelDescriptor, memory: Double = installedGigabytes) -> Double? {
        switch model.family.media {
        case .video: peak(model: model, size: PixelSize(width: 512, height: 512), frames: 25, lowMemory: true, memory: memory)
        case .image: peak(model: model, size: PixelSize(width: 512, height: 512), frames: nil, lowMemory: true, memory: memory)
        }
    }
}
