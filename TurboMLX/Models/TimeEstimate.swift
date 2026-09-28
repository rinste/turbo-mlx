import Foundation

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
        let work = Work(model: model, size: request.size, steps: request.steps, guidance: request.guidance, frames: request.frames)
        let byID = Dictionary(models.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let (rates, _) = rates(for: model, work: work, lowMemory: request.lowMemory, history: history, models: byID)
        return Plan(steps: work.stepUnits.map { $0 * rates.denoise }, decode: rates.decode * work.decode)
    }

    /// `count` generations with these settings, after loading the model if it is not in memory.
    static func estimate(
        model: ModelDescriptor, size: PixelSize, steps: Int, guidance: Double, frames: Int?, lowMemory: Bool,
        count: Int, isLoaded: Bool, history: [HistoryItem], models: [ModelDescriptor]
    ) -> Result {
        let work = Work(model: model, size: size, steps: steps, guidance: guidance, frames: frames)
        let byID = Dictionary(models.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        let (rates, isFromHistory) = rates(for: model, work: work, lowMemory: lowMemory, history: history, models: byID)
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

        init(model: ModelDescriptor, size: PixelSize, steps: Int, guidance: Double, frames: Int?) {
            let pixels = Double(size.width * size.height)
            cfg = model.supportsGuidance && guidance > 1
            if model.family.media == .video {
                let frames = frames ?? 121
                // 32 × 32 pixels and 8 frames per token (the first frame alone).
                let tokens = pixels / 1024 * Double((frames - 1) / 8 + 1)
                stepUnits = Array(repeating: Self.attended(tokens / 4) / 1000, count: max(steps, 0))
                    + Array(repeating: Self.attended(tokens) / 1000, count: 3)
                decode = pixels / 1_000_000 * Double(frames)
            } else {
                // 16 × 16 pixels per token.
                stepUnits = Array(repeating: (cfg ? 2 : 1) * Self.attended(pixels / 256) / 1000, count: max(steps, 0))
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

    /// The native engine on an M1 Max (32 GPU cores), from the generations made while building it.
    private static func reference(_ model: ModelDescriptor, cfg: Bool) -> Rates {
        switch model.family {
        case .flux2Klein: Rates(denoise: 1.86, decode: 1.2, fixed: 0.8, load: 3)
        case .zImageTurbo: Rates(denoise: 2.5, decode: 1, fixed: 0.5, load: 3)
        // With CFG, Ming-Image takes three times as long, not two.
        case .ming: Rates(denoise: cfg ? 4.25 : 2.75, decode: 3.1, fixed: 1.3, load: 5)
        case .qwenImage: Rates(denoise: 2.85, decode: 6.7, fixed: 3.3, load: 5)
        // The 8-bit transformer is slower: 294 s against 231 s for 5 s at 768 × 512.
        case .ltx2 where model.id.contains("q8"): Rates(denoise: 6.93, decode: 0.7, fixed: 12, load: 5)
        case .ltx2: Rates(denoise: 5.3, decode: 0.6, fixed: 12, load: 5)
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

        init?(_ item: HistoryItem, model: ModelDescriptor) {
            let request = item.request
            guard let timings = item.timings, let denoise = timings["denoise"], denoise > 0 else { return nil }
            work = Work(model: model, size: request.size, steps: request.steps, guidance: request.guidance, frames: request.frames)
            guard work.denoise > 0, work.decode > 0 else { return nil }
            self.denoise = denoise
            decode = timings["decode"] ?? 0
            load = timings["load"] ?? 0
            fixed = max(0, item.seconds - denoise - decode - load)
            lowMemory = request.lowMemory
        }
    }

    private static func rates(
        for model: ModelDescriptor, work: Work, lowMemory: Bool, history: [HistoryItem], models: [String: ModelDescriptor]
    ) -> (Rates, Bool) {
        let reference = reference(model, cfg: work.cfg)
        // Recent generations with the same model, preferably with the same CFG and memory saving,
        // and the five closest in size: small images use the GPU less well than large ones.
        var samples: [Sample] = Array(history.lazy.filter { $0.modelID == model.id }.compactMap { Sample($0, model: model) }.prefix(20))
        if samples.contains(where: { $0.work.cfg == work.cfg }) { samples = samples.filter { $0.work.cfg == work.cfg } }
        if samples.contains(where: { $0.lowMemory == lowMemory }) { samples = samples.filter { $0.lowMemory == lowMemory } }
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

    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }
}
