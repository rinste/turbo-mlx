import Foundation
import ImageIO

/// How long a generation should take on this Mac, for the Generate button. It follows the finished
/// generations of the same model in the history; a model not used yet gets the times measured on an
/// M1 Max, scaled by how this Mac compared on the models it did use (or by its chip, before any).
enum TimeEstimate {
    struct Result {
        var seconds: Double
        /// Fitted to this Mac's own generations with the model, not guessed from another Mac.
        var isFromHistory: Bool
    }

    /// A job's expected seconds, step by step and for the decode, to count down from while it
    /// runs: a clip's refining steps cost several times its first ones.
    struct Plan {
        var steps: [Double]
        var decode: Double
    }

    static func plan(model: ModelDescriptor, request: GenerationRequest, history: [HistoryItem], models: [ModelDescriptor]) -> Plan {
        let work = Work(model: model, size: request.size, steps: request.steps, guidance: request.guidance, frames: request.frames,
                        reference: request.referenceImage)
        let byID = Dictionary(models.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let (rates, _) = rates(for: model, work: work, lowMemory: request.lowMemory, halfPrecision: request.halfPrecision ?? false,
                               history: history, models: byID)
        // An upscale's decode rate also covers the picture's encode, done before its step: the
        // decode itself is 72% of the two.
        let decodeShare = model.family.isUpscaler ? 0.72 : 1
        return Plan(steps: work.stepUnits.map { $0 * rates.denoise }, decode: rates.decode * work.decode * decodeShare)
    }

    /// `count` generations with these settings, after loading the model if it is not in memory.
    static func estimate(
        model: ModelDescriptor, size: PixelSize, steps: Int, guidance: Double, frames: Int?, reference: String?, lowMemory: Bool,
        halfPrecision: Bool, count: Int, isLoaded: Bool, history: [HistoryItem], models: [ModelDescriptor]
    ) -> Result {
        let work = Work(model: model, size: size, steps: steps, guidance: guidance, frames: frames, reference: reference)
        let byID = Dictionary(models.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let (rates, isFromHistory) = rates(for: model, work: work, lowMemory: lowMemory, halfPrecision: halfPrecision,
                                           history: history, models: byID)
        let seconds = (isLoaded ? 0 : rates.load) + Double(max(count, 1)) * rates.seconds(for: work)
        return Result(seconds: seconds, isFromHistory: isFromHistory)
    }

    /// What a generation's time grows with.
    private struct Work {
        /// Each step's thousands of tokens, weighted by attention (twice with CFG). A clip's first
        /// stage runs at half size, a quarter of its tokens, then three steps refine it at full size.
        var stepUnits: [Double]
        /// Megapixels, × frames for a clip.
        var decode: Double
        var cfg: Bool

        var denoise: Double { stepUnits.reduce(0, +) }

        init(model: ModelDescriptor, size: PixelSize, steps: Int, guidance: Double, frames: Int?, reference: String?) {
            let pixels = Double(size.width * size.height)
            cfg = model.supportsGuidance && guidance > 1
            if model.family.media == .video {
                let frames = frames ?? 121
                // 32 × 32 pixels and 8 frames per token (the first frame alone).
                let tokens = pixels / 1024 * Double((frames - 1) / 8 + 1)
                stepUnits = Array(repeating: Self.attended(tokens / 4) / 1000, count: max(steps, 0))
                    + Array(repeating: Self.attended(tokens) / 1000, count: 3)
                decode = pixels / 1_000_000 * Double(frames)
            } else if model.family.isUpscaler {
                // One step over the result's tokens, in windows of a few hundred: no attention
                // term. Encoding the picture (at the result's size) counts with the decode.
                stepUnits = [pixels / 256 / 1000]
                decode = pixels / 1_000_000
            } else {
                // 16 × 16 pixels per token; FLUX.2 Klein reads a reference image's tokens in every pass
                // too, and Qwen-Image Edit its picture's, encoded at about the image's size.
                let referenceTokens = switch model.family {
                case .flux2Klein: Double(TimeEstimate.referenceTokens(reference))
                case .qwenImageEdit: pixels / 256
                default: 0.0
                }
                stepUnits = Array(repeating: (cfg ? 2 : 1) * Self.attended(pixels / 256 + referenceTokens) / 1000, count: max(steps, 0))
                decode = pixels / 1_000_000
            }
        }

        /// Attention makes a token dearer the more tokens there are: at about 25,000 it costs as
        /// much as the rest of a block.
        private static func attended(_ tokens: Double) -> Double {
            tokens * (1 + tokens / 25_000)
        }
    }

    /// Seconds per unit of `Work`, plus the parts that do not grow with it.
    private struct Rates {
        var denoise: Double
        var decode: Double
        /// Reading the prompt and saving the result.
        var fixed: Double
        /// Loading the model, when it is not in memory yet.
        var load: Double

        func seconds(for work: Work) -> Double {
            fixed + denoise * work.denoise + decode * work.decode
        }
    }

    /// The native engine on an M1 Max (32 GPU cores), from the generations made while building it,
    /// scaled on 29 September 2026 by what mlx-swift 0.32.2 changed in the same requests (Klein's
    /// and Ming's steps 16% faster, the Qwen-Image and Ming decoders twice as fast, LTX's decoder
    /// too, SeedVR2's encode and decode two to three times).
    private static func reference(_ model: ModelDescriptor, cfg: Bool) -> Rates {
        switch model.family {
        case .flux2Klein: Rates(denoise: 1.56, decode: 1.2, fixed: 0.8, load: 3)
        case .zImageTurbo: Rates(denoise: 2.5, decode: 1, fixed: 0.5, load: 3)
        // With CFG, Ming-Image takes three times as long, not two.
        case .ming: Rates(denoise: cfg ? 3.55 : 2.3, decode: 2, fixed: 1.3, load: 5)
        case .qwenImage: Rates(denoise: 2.5, decode: 3.4, fixed: 3.3, load: 5)
        // Qwen-Image's transformer over the image's and the picture's tokens: 745 s for 672 × 880 in
        // 20 steps (mflux; the native engine matched it at 336 × 432), reading the picture ~4 s.
        case .qwenImageEdit: Rates(denoise: 3.2, decode: 3.4, fixed: 5, load: 5)
        // 8 steps at 512 / 1024 / 2048 px in 7, 27–31 and 150 s; the pixel head is part of each
        // step, so there is no decode to speak of. An edit reads its picture with the prompt
        // (5 s at 1024 px, 17 s at 2048), which the history's fixed part learns.
        case .senseNova: Rates(denoise: 0.72, decode: 0.05, fixed: 1.3, load: 3)
        // The 8-bit transformer is as fast as the 4-bit one since mlx-swift 0.32.2: 5 s at
        // 768 × 512 in 219 s without Save memory, against 206 s with it for the 4-bit one.
        case .ltx2 where model.id.contains("q8"): Rates(denoise: 5.4, decode: 0.28, fixed: 12, load: 5)
        case .ltx2: Rates(denoise: 5.3, decode: 0.28, fixed: 12, load: 5)
        // Not measured yet: LTX-2.3's rates, the same transformer and decoder.
        case .ltx25: Rates(denoise: 5.3, decode: 0.28, fixed: 12, load: 5)
        // 1280 × 1024 in 10 s, 2304 × 2304 in 41 s, 4096 × 4096 in 125 s (encode and decode 5, 20
        // and 62 s of it).
        case .seedVR2: Rates(denoise: 0.97, decode: 4, fixed: 1, load: 3)
        }
    }

    /// A finished generation's times split along `Work`, or nil when it has no timings.
    private struct Sample {
        var work: Work
        var denoise: Double
        var decode: Double
        var fixed: Double
        var load: Double
        var lowMemory: Bool
        var halfPrecision: Bool

        init?(_ item: HistoryItem, model: ModelDescriptor) {
            let request = item.request
            guard let timings = item.timings, let denoise = timings["denoise"], denoise > 0 else { return nil }
            work = Work(model: model, size: request.size, steps: request.steps, guidance: request.guidance, frames: request.frames,
                        reference: request.referenceImage)
            guard work.denoise > 0, work.decode > 0 else { return nil }
            self.denoise = denoise
            // An upscale's encode is the picture's, at the result's size: it counts with the decode.
            decode = (timings["decode"] ?? 0) + (model.family.isUpscaler ? timings["encode"] ?? 0 : 0)
            load = timings["load"] ?? 0
            fixed = max(0, item.seconds - denoise - decode - load)
            lowMemory = request.lowMemory
            halfPrecision = request.halfPrecision ?? false
        }
    }

    private static func rates(
        for model: ModelDescriptor, work: Work, lowMemory: Bool, halfPrecision: Bool, history: [HistoryItem], models: [String: ModelDescriptor]
    ) -> (Rates, Bool) {
        let reference = reference(model, cfg: work.cfg)
        // Recent generations with the same model, preferably with the same CFG, memory saving and
        // precision, and the five closest in size: small images use the GPU less well than large ones.
        var samples: [Sample] = Array(history.lazy.filter { $0.modelID == model.id }.compactMap { Sample($0, model: model) }.prefix(20))
        if samples.contains(where: { $0.work.cfg == work.cfg }) { samples = samples.filter { $0.work.cfg == work.cfg } }
        if samples.contains(where: { $0.lowMemory == lowMemory }) { samples = samples.filter { $0.lowMemory == lowMemory } }
        if samples.contains(where: { $0.halfPrecision == halfPrecision }) { samples = samples.filter { $0.halfPrecision == halfPrecision } }
        samples = Array(samples.sorted { abs(log($0.work.denoise / work.denoise)) < abs(log($1.work.denoise / work.denoise)) }.prefix(5))
        if !samples.isEmpty {
            let rates = Rates(
                denoise: median(samples.map { $0.denoise / $0.work.denoise }) ?? reference.denoise,
                decode: median(samples.map { $0.decode / $0.work.decode }) ?? reference.decode,
                fixed: median(samples.map(\.fixed)) ?? reference.fixed,
                load: median(samples.compactMap { $0.load > 0 ? $0.load : nil }) ?? reference.load
            )
            return (rates, true)
        }
        let factor = machineFactor(history: history, models: models)
        return (Rates(denoise: reference.denoise * factor, decode: reference.decode * factor,
                      fixed: reference.fixed * factor, load: reference.load), false)
    }

    /// This Mac's denoising time against the M1 Max's, on the models it used; before it used any,
    /// a guess from its chip.
    private static func machineFactor(history: [HistoryItem], models: [String: ModelDescriptor]) -> Double {
        let ratios = history.prefix(40).compactMap { item -> Double? in
            guard let model = models[item.modelID], let sample = Sample(item, model: model) else { return nil }
            return sample.denoise / (reference(model, cfg: sample.work.cfg).denoise * sample.work.denoise)
        }
        return median(ratios) ?? chipFactor
    }

    /// Time against an M1 Max from the chip's name ("Apple M3 Pro"): the size of its GPU (base,
    /// Pro, Max, Ultra) and how much faster each generation's cores are. Rough, and replaced by
    /// what this Mac measures from its first image on.
    private static let chipFactor: Double = {
        var length = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &length, nil, 0)
        var buffer = [CChar](repeating: 0, count: max(length, 1))
        sysctlbyname("machdep.cpu.brand_string", &buffer, &length, nil, 0)
        let name = String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        let size = name.contains("Ultra") ? 2.0 : name.contains("Max") ? 1.0 : name.contains("Pro") ? 0.55 : 0.3
        let generation = name.firstMatch(of: /M(\d+)/).flatMap { Int($0.1) } ?? 1
        let perCore = [1: 1.0, 2: 1.15, 3: 1.3, 4: 1.5][generation] ?? 2.0
        return 1 / (size * perCore)
    }()

    /// The tokens a reference image adds to each of FLUX.2 Klein's passes, as the engine encodes it
    /// (`KleinReference`): in its own proportions at most about a megapixel, sides cut to multiples of
    /// 16, a token per 16 × 16 pixels. Read from the file's header once per picture.
    static func referenceTokens(_ name: String?) -> Int {
        guard let name else { return 0 }
        if let known = referenceTokenCounts[name] { return known }
        var tokens = 0
        if let source = CGImageSourceCreateWithURL(HistoryStore.referenceURL(name) as CFURL, nil),
           let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int {
            let scale = min(1, (1_048_576 / Double(width * height)).squareRoot())
            tokens = (Int((Double(width) * scale).rounded()) / 16) * (Int((Double(height) * scale).rounded()) / 16)
        }
        referenceTokenCounts[name] = tokens
        return tokens
    }

    private static var referenceTokenCounts: [String: Int] = [:]

    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }
}
