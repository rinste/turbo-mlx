import Foundation
import ImageIO
import Synchronization

nonisolated struct PixelSize: Hashable, Codable, Sendable {
    var width: Int
    var height: Int

    var megapixels: Double { Double(width * height) / 1_000_000 }

    /// Ming-Image (like most DiTs) needs both sides to be multiples of 16; LTX-2's two stages, 64.
    static func snapped(_ value: Double, multiple: Int = 16) -> Int {
        max(256, min(4096, Int((value / Double(multiple)).rounded()) * multiple))
    }
}

nonisolated enum AspectRatio: String, CaseIterable, Codable, Identifiable, Sendable {
    case square = "1:1"
    case portrait4x5 = "4:5"
    case portrait3x4 = "3:4"
    case portrait2x3 = "2:3"
    case portrait9x16 = "9:16"
    case landscape4x3 = "4:3"
    case landscape3x2 = "3:2"
    case landscape16x9 = "16:9"

    var id: String { rawValue }

    var ratio: Double {
        let parts = rawValue.split(separator: ":").compactMap { Double($0) }
        return parts[0] / parts[1]
    }

    var usage: String {
        switch self {
        case .square: "Square"
        case .portrait4x5: "Social, portrait"
        case .portrait3x4: "Portrait"
        case .portrait2x3: "Poster"
        case .portrait9x16: "Stories, phone"
        case .landscape4x3: "Landscape"
        case .landscape3x2: "Card, photo"
        case .landscape16x9: "Banner, screen"
        }
    }

    /// Keeps roughly `base`² pixels, the way the model's aspect-ratio buckets were built.
    func size(base: Int, multiple: Int = 16) -> PixelSize {
        let area = Double(base * base)
        return PixelSize(
            width: PixelSize.snapped((area * ratio).squareRoot(), multiple: multiple),
            height: PixelSize.snapped((area / ratio).squareRoot(), multiple: multiple)
        )
    }
}

/// One piece of the prompt. The blocks are joined, in order, into the text sent to the model, so
/// the subject, the style or the lighting can each live in a block of their own and be moved around.
nonisolated struct PromptBlock: Codable, Hashable, Identifiable, Sendable {
    static let subjectName = "Subject"
    static let styleName = "Style"

    var id = UUID()
    var name: String
    var text = ""
    /// How tall the text is on screen, once the user has resized the block.
    var height: Double?

    /// A new prompt: the subject, then the style, each in a block of its own.
    static func defaults(subject: String = "") -> [PromptBlock] {
        [PromptBlock(name: subjectName, text: subject), PromptBlock(name: styleName)]
    }

    /// The blocks as one prompt: a comma between pieces, or just a space after punctuation.
    static func joined(_ blocks: [PromptBlock]) -> String {
        var result = ""
        for block in blocks {
            let piece = block.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !piece.isEmpty else { continue }
            if result.isEmpty {
                result = piece
            } else if let last = result.last, ".,;:!?".contains(last) {
                result += " " + piece
            } else {
                result += ", " + piece
            }
        }
        return result
    }

    /// "Prompt N" with the first N not already used.
    static func defaultName(among blocks: [PromptBlock]) -> String {
        let taken = Set(blocks.map(\.name))
        var number = blocks.count + 1
        while taken.contains("Prompt \(number)") { number += 1 }
        return "Prompt \(number)"
    }
}

nonisolated struct GenerationSettings: Codable, Equatable, Sendable {
    static let resolutions = [512, 768, 1024, 1536, 2048]
    /// Video: 640 gives 768 × 512 at 3:2 and 768 gives 1024 × 576 at 16:9, the sizes LTX-2 is at ease with.
    static let videoResolutions = [512, 640, 768]
    static let videoDurations = 1.0...10.0
    static let videoFrameRates = [24, 25, 30]
    static let guidanceRange = 1.0...7.0
    static let maxBatch = 8
    /// How many times an upscaler enlarges the picture's sides.
    static let upscaleFactors: [Double] = [2, 3, 4]

    var blocks = PromptBlock.defaults()
    var aspect = AspectRatio.square
    var resolution = 768
    var usesCustomSize = false
    var customWidth = 1024
    var customHeight = 1024
    var steps = 12
    var guidance = 1.0
    var randomSeed = true
    var seed = 42
    var batchCount = 1
    var transparentBackground = true
    var lowMemory = false
    var videoResolution = 640
    var videoSeconds = 5.0
    var videoFrameRate = 24
    /// A model that picks the length from the prompt does so, `videoSeconds` being the longest.
    var videoAutoDuration = false
    /// The image a clip starts from, or an image is made from: a file in the references folder
    /// (`HistoryStore`).
    var referenceImage: String?
    /// Upscalers: the factor, and how much the picture is softened before it is enlarged (0–1),
    /// which gives smoother results from a noisy or over-sharpened picture.
    var upscale = 2.0
    var softness = 0.0

    var size: PixelSize {
        usesCustomSize
            ? PixelSize(width: PixelSize.snapped(Double(customWidth)), height: PixelSize.snapped(Double(customHeight)))
            : aspect.size(base: resolution)
    }

    /// A clip's size: sides multiples of 64.
    var videoSize: PixelSize {
        usesCustomSize
            ? PixelSize(width: PixelSize.snapped(Double(customWidth), multiple: 64), height: PixelSize.snapped(Double(customHeight), multiple: 64))
            : aspect.size(base: videoResolution, multiple: 64)
    }

    func size(for family: ModelFamily) -> PixelSize {
        if family.isUpscaler {
            return upscaledSize ?? PixelSize(width: Int(1024 * upscale), height: Int(1024 * upscale))
        }
        if family.media == .video { return videoSize }
        let multiple = family.sizeMultiple
        guard multiple != 16 else { return size }
        return usesCustomSize
            ? PixelSize(width: PixelSize.snapped(Double(customWidth), multiple: multiple),
                        height: PixelSize.snapped(Double(customHeight), multiple: multiple))
            : aspect.size(base: resolution, multiple: multiple)
    }

    /// What an upscaler makes of the reference picture, nil without one.
    var upscaledSize: PixelSize? {
        referenceImage.flatMap(ReferencePicture.size).map { Upscale.outputSize(of: $0, factor: upscale) }
    }

    /// The clip's frames: 8k + 1 (LTX-2's latent frames cover eight), nearest to the duration.
    var videoFrames: Int {
        Int((videoSeconds * Double(videoFrameRate) / 8).rounded()) * 8 + 1
    }

    /// Whether this family picks the clip's length, as the settings ask it to.
    func autoDuration(for family: ModelFamily) -> Bool {
        videoAutoDuration && family.picksDuration
    }

    /// The prompt sent to the model: the blocks joined in order, already trimmed.
    var prompt: String { PromptBlock.joined(blocks) }
    var trimmedPrompt: String { prompt }

    init() {}

    private enum CodingKeys: String, CodingKey {
        case blocks, aspect, resolution, usesCustomSize, customWidth, customHeight, steps, guidance,
             randomSeed, seed, batchCount, transparentBackground, lowMemory,
             videoResolution, videoSeconds, videoFrameRate, videoAutoDuration, referenceImage, upscale, softness
    }

    /// Settings saved before the prompt had blocks kept a single string.
    private enum LegacyKeys: String, CodingKey {
        case prompt
    }

    init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        if let saved = try values.decodeIfPresent([PromptBlock].self, forKey: .blocks), !saved.isEmpty {
            blocks = saved
        } else {
            let legacy = try decoder.container(keyedBy: LegacyKeys.self)
            blocks = PromptBlock.defaults(subject: try legacy.decodeIfPresent(String.self, forKey: .prompt) ?? "")
        }
        aspect = try values.decodeIfPresent(AspectRatio.self, forKey: .aspect) ?? aspect
        resolution = try values.decodeIfPresent(Int.self, forKey: .resolution) ?? resolution
        usesCustomSize = try values.decodeIfPresent(Bool.self, forKey: .usesCustomSize) ?? usesCustomSize
        customWidth = try values.decodeIfPresent(Int.self, forKey: .customWidth) ?? customWidth
        customHeight = try values.decodeIfPresent(Int.self, forKey: .customHeight) ?? customHeight
        steps = try values.decodeIfPresent(Int.self, forKey: .steps) ?? steps
        guidance = try values.decodeIfPresent(Double.self, forKey: .guidance) ?? guidance
        randomSeed = try values.decodeIfPresent(Bool.self, forKey: .randomSeed) ?? randomSeed
        seed = try values.decodeIfPresent(Int.self, forKey: .seed) ?? seed
        batchCount = try values.decodeIfPresent(Int.self, forKey: .batchCount) ?? batchCount
        transparentBackground = try values.decodeIfPresent(Bool.self, forKey: .transparentBackground) ?? transparentBackground
        lowMemory = try values.decodeIfPresent(Bool.self, forKey: .lowMemory) ?? lowMemory
        videoResolution = try values.decodeIfPresent(Int.self, forKey: .videoResolution) ?? videoResolution
        videoSeconds = try values.decodeIfPresent(Double.self, forKey: .videoSeconds) ?? videoSeconds
        videoFrameRate = try values.decodeIfPresent(Int.self, forKey: .videoFrameRate) ?? videoFrameRate
        videoAutoDuration = try values.decodeIfPresent(Bool.self, forKey: .videoAutoDuration) ?? videoAutoDuration
        referenceImage = try values.decodeIfPresent(String.self, forKey: .referenceImage)
        upscale = try values.decodeIfPresent(Double.self, forKey: .upscale) ?? upscale
        softness = try values.decodeIfPresent(Double.self, forKey: .softness) ?? softness
    }

    /// Seeds for the next batch: consecutive from the fixed seed, or fresh random ones.
    func nextSeeds() -> [Int] {
        let count = max(1, min(Self.maxBatch, batchCount))
        if randomSeed {
            return (0..<count).map { _ in Int.random(in: 0..<1_000_000_000) }
        }
        return (0..<count).map { seed + $0 }
    }

    /// Picks the preset that produced `size`, or switches to a custom size.
    mutating func apply(size: PixelSize, video: Bool = false, multiple: Int = 16) {
        for aspect in AspectRatio.allCases {
            for resolution in video ? Self.videoResolutions : Self.resolutions
            where aspect.size(base: resolution, multiple: video ? 64 : multiple) == size {
                self.aspect = aspect
                if video { videoResolution = resolution } else { self.resolution = resolution }
                usesCustomSize = false
                return
            }
        }
        usesCustomSize = true
        customWidth = size.width
        customHeight = size.height
    }
}

/// Everything the backend needs for one image.
nonisolated struct GenerationRequest: Codable, Hashable, Sendable {
    var prompt: String
    /// The blocks the prompt was assembled from, so Reuse Prompt and Settings brings them back
    /// (older history items have none).
    var blocks: [PromptBlock]?
    var seed: Int
    var size: PixelSize
    var steps: Int
    var guidance: Double
    var transparentBackground: Bool
    var lowMemory: Bool
    /// Video only: how many frames and at what rate; nil for an image. With `autoDuration` the
    /// model picks the length, `frames` being the longest until the engine says which it chose.
    var frames: Int?
    var fps: Int?
    var autoDuration: Bool?
    /// Video only: the image the clip starts from, a file in the references folder.
    var referenceImage: String?
    /// Upscalers only: the factor and the softening (`GenerationSettings`).
    var upscale: Double?
    var softness: Double?

    /// What to show for it: the prompt, or for an upscale, what it did ("Upscaled 2×").
    var caption: String {
        guard let upscale else { return prompt }
        let factor = upscale.rounded() == upscale ? "\(Int(upscale))" : upscale.formatted(.number.precision(.fractionLength(1)))
        let softened = (softness ?? 0) > 0 ? ", softened \(Int(((softness ?? 0) * 100).rounded()))%" : ""
        return "Upscaled \(factor)×\(softened)"
    }
}

/// The size an upscale comes out at, as the engine computes it (mflux's `ScaleFactor` and
/// `SeedVR2Util.preprocess_image`): the shorter side times the factor, cut down to a multiple of
/// 16, the other side in proportion, both even.
nonisolated enum Upscale {
    /// The largest result: 4096 × 4096 takes about 2 minutes on an M1 Max and peaks at 18 GB.
    static let maxMegapixels = 16.8

    static func outputSize(of picture: PixelSize, factor: Double) -> PixelSize {
        let shorter = Double(min(picture.width, picture.height))
        let product = factor * shorter
        let scale = (product - product.truncatingRemainder(dividingBy: 16)).rounded(.towardZero) / shorter
        return PixelSize(width: Int(Double(picture.width) * scale) / 2 * 2, height: Int(Double(picture.height) * scale) / 2 * 2)
    }
}

/// The pixel size of a picture in the references folder, read from its header once.
nonisolated enum ReferencePicture {
    private static let sizes = Mutex<[String: PixelSize]>([:])

    static func size(_ name: String) -> PixelSize? {
        if let known = sizes.withLock({ $0[name] }) { return known }
        guard let source = CGImageSourceCreateWithURL(HistoryStore.referenceURL(name) as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              var width = properties[kCGImagePropertyPixelWidth] as? Int, var height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        // Turned upright, as the engine reads it.
        if let orientation = properties[kCGImagePropertyOrientation] as? Int, orientation >= 5 { swap(&width, &height) }
        let size = PixelSize(width: width, height: height)
        sizes.withLock { $0[name] = size }
        return size
    }
}
