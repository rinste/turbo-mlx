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
    /// A song's longest length, in seconds: MiniMax-Music3 makes up to six minutes, and may end sooner.
    static let songDurations = 10.0...360.0
    /// A sound effect's length, in seconds: MOSS-SoundEffect makes up to 30, in tenths.
    static let soundDurations = 0.5...30.0
    /// How much an image may change from the picture it starts from: at least one step is always
    /// skipped and run, so the ends would only give the picture back or ignore it.
    static let imageChangeRange = 0.05...0.95

    var blocks = PromptBlock.defaults()
    /// What the image should not have, for the models that take it (`supportsNegativePrompt`).
    var negativePrompt = ""
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
    /// The image a clip ends on (`ModelFamily.takesLastFrame`), in the same folder.
    var lastFrameImage: String?
    /// Image to image (`ModelFamily.startsFromImage`): how much the new image may change from the
    /// reference picture, as A1111 and InvokeAI's denoising strength (0.75 by default there too).
    /// The engine takes the part kept, mflux's image strength (`imageStrength(for:)`).
    var imageChange = 0.75
    /// FLUX.2 Klein, which can do either: start from the reference picture instead of editing it.
    var startsFromReference = false
    /// Upscalers: the factor, and how much the picture is softened before it is enlarged (0–1),
    /// which gives smoother results from a noisy or over-sharpened picture.
    var upscale = 2.0
    var softness = 0.0
    /// The transformer's activations in 16 bits where the reference keeps 32 (Z-Image, Qwen-Image
    /// and its editor; the other families run in 16 bits already): faster, slightly different
    /// fine detail. Off, the image is the reference's.
    var halfPrecision = false
    /// The image shown as it forms, every few steps, while it is generated.
    var livePreview = true
    /// LTX: the full pipeline (Quality) instead of the distilled one (Fast).
    var videoQuality = false
    /// Idol: how many Gaussians the object is made of (its detail), and what kind of picture it comes from.
    var gaussians = ObjectDetail.defaultGaussians
    var objectMode = ObjectMode.render
    /// A song's lyrics, section tags such as "[verse]" on lines of their own, and the longest it may last (the prompt
    /// describes the music). An instrumental song keeps the lyrics aside and gets `instrumentalTag` instead.
    var lyrics = ""
    var instrumental = false
    var songSeconds = 60.0
    /// A sound effect's length (the prompt describes the sound).
    var soundSeconds = 10.0
    /// The LoRA chosen for each family that takes them (`ModelFamily.takesLoRAs`), by its raw
    /// value: a LoRA fits one family, so each keeps its own.
    var loras: [String: LoRAChoice] = [:]

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
        if family.media == .model3D || family.media == .audio { return PixelSize(width: 0, height: 0) }
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

    /// Whether `model` runs its Quality mode with these settings.
    func fullPipeline(for model: ModelDescriptor) -> Bool {
        videoQuality && model.family.isLTX
    }

    /// Whether an image of `family` starts from the reference picture rather than editing it: always
    /// for the families that only start from one, for FLUX.2 Klein when asked to.
    func startsFromImage(for family: ModelFamily) -> Bool {
        family.startsFromImage && (startsFromReference || !family.editsReference)
    }

    /// What the engine keeps of the reference picture (`GenerationRequest.imageStrength`), in
    /// hundredths: nil without a picture, or when `family` edits it instead.
    func imageStrength(for family: ModelFamily) -> Double? {
        guard referenceImage != nil, startsFromImage(for: family) else { return nil }
        return ((1 - imageChange) * 100).rounded() / 100
    }

    func lora(for family: ModelFamily) -> LoRAChoice? {
        family.takesLoRAs ? loras[family.rawValue] : nil
    }

    mutating func setLoRA(_ choice: LoRAChoice?, for family: ModelFamily) {
        loras[family.rawValue] = choice
    }

    /// The prompt sent to the model: the blocks joined in order, already trimmed.
    var prompt: String { PromptBlock.joined(blocks) }
    var trimmedPrompt: String { prompt }
    /// MiniMax-Music3's section tag for a song without words (the model card's only way to ask for one).
    static let instrumentalTag = "[instrumental]"

    /// The lyrics as the song gets them: the instrumental tag, or the lyrics without the blank lines around them.
    var songLyrics: String {
        instrumental ? Self.instrumentalTag : lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The negative prompt `model` would get: nil where it has none, or guidance is off and there
    /// is no unconditional pass to use it.
    func negativePrompt(for model: ModelDescriptor) -> String? {
        let text = negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        // LTX's Quality mode always runs its sound's guidance, which the negative prompt steers too.
        if fullPipeline(for: model) { return text }
        return model.supportsNegativePrompt && guidance > 1 ? text : nil
    }

    init() {}

    private enum CodingKeys: String, CodingKey {
        case blocks, negativePrompt, aspect, resolution, usesCustomSize, customWidth, customHeight, steps, guidance,
             randomSeed, seed, batchCount, transparentBackground, lowMemory,
             videoResolution, videoSeconds, videoFrameRate, videoAutoDuration, referenceImage, lastFrameImage, upscale, softness,
             halfPrecision, livePreview, loras, videoQuality, gaussians, objectMode, lyrics, instrumental, songSeconds, soundSeconds,
             imageChange, startsFromReference
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
        negativePrompt = try values.decodeIfPresent(String.self, forKey: .negativePrompt) ?? negativePrompt
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
        lastFrameImage = try values.decodeIfPresent(String.self, forKey: .lastFrameImage)
        upscale = try values.decodeIfPresent(Double.self, forKey: .upscale) ?? upscale
        softness = try values.decodeIfPresent(Double.self, forKey: .softness) ?? softness
        halfPrecision = try values.decodeIfPresent(Bool.self, forKey: .halfPrecision) ?? halfPrecision
        livePreview = try values.decodeIfPresent(Bool.self, forKey: .livePreview) ?? livePreview
        loras = try values.decodeIfPresent([String: LoRAChoice].self, forKey: .loras) ?? loras
        videoQuality = try values.decodeIfPresent(Bool.self, forKey: .videoQuality) ?? videoQuality
        gaussians = try values.decodeIfPresent(Int.self, forKey: .gaussians) ?? gaussians
        objectMode = try values.decodeIfPresent(ObjectMode.self, forKey: .objectMode) ?? objectMode
        lyrics = try values.decodeIfPresent(String.self, forKey: .lyrics) ?? lyrics
        instrumental = try values.decodeIfPresent(Bool.self, forKey: .instrumental) ?? instrumental
        songSeconds = try values.decodeIfPresent(Double.self, forKey: .songSeconds) ?? songSeconds
        soundSeconds = try values.decodeIfPresent(Double.self, forKey: .soundSeconds) ?? soundSeconds
        imageChange = try values.decodeIfPresent(Double.self, forKey: .imageChange) ?? imageChange
        startsFromReference = try values.decodeIfPresent(Bool.self, forKey: .startsFromReference) ?? startsFromReference
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

/// Idol's detail: the Gaussians an object is made of, 64K to 1M (the engine takes 32,768 to 1,048,576, multiples of 32).
nonisolated enum ObjectDetail {
    static let defaultGaussians = 262_144
    static let choices: [(gaussians: Int, label: String)] = [
        (65_536, "64K"), (131_072, "128K"), (262_144, "256K"), (524_288, "512K"), (1_048_576, "1M"),
    ]

    static func label(_ gaussians: Int) -> String {
        choices.first { $0.gaussians == gaussians }?.label ?? "\(gaussians / 1024)K"
    }
}

/// What kind of picture Idol is given, which sets how tightly it is cropped around the object.
nonisolated enum ObjectMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// A render or a generated picture: idol-sdk's "synthetic", cropped tight.
    case render = "synthetic"
    /// A photo: "real", cropped looser.
    case photo = "real"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .render: "Render"
        case .photo: "Photo"
        }
    }
}

/// Everything the backend needs for one image.
nonisolated struct GenerationRequest: Codable, Hashable, Sendable {
    var prompt: String
    /// The blocks the prompt was assembled from, so Reuse Prompt and Settings brings them back
    /// (older history items have none).
    var blocks: [PromptBlock]?
    /// What guidance steered away from (`GenerationSettings.negativePrompt(for:)`); nil without one.
    var negativePrompt: String?
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
    /// LTX only: the image the clip ends on, in the same folder.
    var lastFrameImage: String?
    /// Image to image: how much of `referenceImage` the image kept (mflux's image strength, one minus
    /// the change asked); nil when the picture was edited from, or there was none.
    var imageStrength: Double?
    /// Upscalers only: the factor and the softening (`GenerationSettings`).
    var upscale: Double?
    var softness: Double?
    /// The 16-bit precision option (`GenerationSettings.halfPrecision`), for the families it
    /// applies to; nil for the others, and in items from before it.
    var halfPrecision: Bool?
    /// The LoRAs applied (files of the LoRA library); nil without any.
    var loras: [LoRAChoice]?
    /// LTX's Quality mode (the full pipeline); nil for Fast, and for the other families.
    var quality: Bool?
    /// A 3D object (Idol): its Gaussians and its picture's kind ("synthetic" or "real"); nil for the others. Such a
    /// request is saved without a size: an object has none.
    var numGaussians: Int?
    var objectMode: String?
    /// A song (MiniMax-Music3): its lyrics and its length in seconds, the longest asked until it is made, then the
    /// length it came out at; a sound effect (MOSS-SoundEffect): no lyrics, and its length. Nil for the others. Such a
    /// request is saved without a size.
    var lyrics: String?
    var duration: Double?

    /// What to show for it: the prompt, for an upscale what it did ("Upscaled 2×"), for an object its detail.
    var caption: String {
        if let numGaussians { return "3D object, \(ObjectDetail.label(numGaussians)) Gaussians" }
        if lyrics != nil { return prompt }
        guard let upscale else { return prompt }
        let factor = upscale.rounded() == upscale ? "\(Int(upscale))" : upscale.formatted(.number.precision(.fractionLength(1)))
        let softened = (softness ?? 0) > 0 ? ", softened \(Int(((softness ?? 0) * 100).rounded()))%" : ""
        return "Upscaled \(factor)×\(softened)"
    }

    /// The steps the engine runs: an image that starts from a picture skips the first ones.
    var denoisingSteps: Int { Self.denoisingSteps(steps, imageStrength: referenceImage == nil ? nil : imageStrength) }

    /// The reference picture, unless the image started from it: an edit reads it in every step, a
    /// start only once.
    var editedPicture: String? { imageStrength == nil ? referenceImage : nil }

    /// `steps` less those an image that starts from a picture skips: mflux's `init_time_step`,
    /// `max(1, floor(steps · strength))` (`ImageToImage.startStep` in the engine).
    static func denoisingSteps(_ steps: Int, imageStrength: Double?) -> Int {
        guard let imageStrength, imageStrength > 0 else { return steps }
        return steps - max(1, Int(Double(steps) * min(1, imageStrength)))
    }
}

nonisolated extension GenerationRequest {
    private enum CodingKeys: String, CodingKey {
        case prompt, blocks, negativePrompt, seed, size, steps, guidance, transparentBackground, lowMemory, frames, fps,
             autoDuration, referenceImage, lastFrameImage, imageStrength, upscale, softness, halfPrecision, loras, quality,
             numGaussians, objectMode, lyrics, duration
    }

    init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        prompt = try values.decode(String.self, forKey: .prompt)
        blocks = try values.decodeIfPresent([PromptBlock].self, forKey: .blocks)
        negativePrompt = try values.decodeIfPresent(String.self, forKey: .negativePrompt)
        seed = try values.decode(Int.self, forKey: .seed)
        // A 3D object and a sound have no size.
        size = try values.decodeIfPresent(PixelSize.self, forKey: .size) ?? PixelSize(width: 0, height: 0)
        steps = try values.decode(Int.self, forKey: .steps)
        guidance = try values.decode(Double.self, forKey: .guidance)
        transparentBackground = try values.decode(Bool.self, forKey: .transparentBackground)
        lowMemory = try values.decode(Bool.self, forKey: .lowMemory)
        frames = try values.decodeIfPresent(Int.self, forKey: .frames)
        fps = try values.decodeIfPresent(Int.self, forKey: .fps)
        autoDuration = try values.decodeIfPresent(Bool.self, forKey: .autoDuration)
        referenceImage = try values.decodeIfPresent(String.self, forKey: .referenceImage)
        lastFrameImage = try values.decodeIfPresent(String.self, forKey: .lastFrameImage)
        imageStrength = try values.decodeIfPresent(Double.self, forKey: .imageStrength)
        upscale = try values.decodeIfPresent(Double.self, forKey: .upscale)
        softness = try values.decodeIfPresent(Double.self, forKey: .softness)
        halfPrecision = try values.decodeIfPresent(Bool.self, forKey: .halfPrecision)
        loras = try values.decodeIfPresent([LoRAChoice].self, forKey: .loras)
        quality = try values.decodeIfPresent(Bool.self, forKey: .quality)
        numGaussians = try values.decodeIfPresent(Int.self, forKey: .numGaussians)
        objectMode = try values.decodeIfPresent(String.self, forKey: .objectMode)
        lyrics = try values.decodeIfPresent(String.self, forKey: .lyrics)
        duration = try values.decodeIfPresent(Double.self, forKey: .duration)
    }

    func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(prompt, forKey: .prompt)
        try values.encodeIfPresent(blocks, forKey: .blocks)
        try values.encodeIfPresent(negativePrompt, forKey: .negativePrompt)
        try values.encode(seed, forKey: .seed)
        if numGaussians == nil, lyrics == nil, duration == nil { try values.encode(size, forKey: .size) }
        try values.encode(steps, forKey: .steps)
        try values.encode(guidance, forKey: .guidance)
        try values.encode(transparentBackground, forKey: .transparentBackground)
        try values.encode(lowMemory, forKey: .lowMemory)
        try values.encodeIfPresent(frames, forKey: .frames)
        try values.encodeIfPresent(fps, forKey: .fps)
        try values.encodeIfPresent(autoDuration, forKey: .autoDuration)
        try values.encodeIfPresent(referenceImage, forKey: .referenceImage)
        try values.encodeIfPresent(lastFrameImage, forKey: .lastFrameImage)
        try values.encodeIfPresent(imageStrength, forKey: .imageStrength)
        try values.encodeIfPresent(upscale, forKey: .upscale)
        try values.encodeIfPresent(softness, forKey: .softness)
        try values.encodeIfPresent(halfPrecision, forKey: .halfPrecision)
        try values.encodeIfPresent(loras, forKey: .loras)
        try values.encodeIfPresent(quality, forKey: .quality)
        try values.encodeIfPresent(numGaussians, forKey: .numGaussians)
        try values.encodeIfPresent(objectMode, forKey: .objectMode)
        try values.encodeIfPresent(lyrics, forKey: .lyrics)
        try values.encodeIfPresent(duration, forKey: .duration)
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
