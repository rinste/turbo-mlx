import Foundation

nonisolated struct PixelSize: Hashable, Codable, Sendable {
    var width: Int
    var height: Int

    var megapixels: Double { Double(width * height) / 1_000_000 }

    /// Ming-Image (like most DiTs) needs both sides to be multiples of 16.
    static func snapped(_ value: Double) -> Int {
        max(256, min(4096, Int((value / 16).rounded()) * 16))
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
    func size(base: Int) -> PixelSize {
        let area = Double(base * base)
        return PixelSize(
            width: PixelSize.snapped((area * ratio).squareRoot()),
            height: PixelSize.snapped((area / ratio).squareRoot())
        )
    }
}

nonisolated struct GenerationSettings: Codable, Equatable, Sendable {
    static let resolutions = [512, 768, 1024, 1536, 2048]
    static let guidanceRange = 1.0...7.0
    static let maxBatch = 8

    var prompt = ""
    var aspect = AspectRatio.square
    var resolution = 1024
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

    var size: PixelSize {
        usesCustomSize
            ? PixelSize(width: PixelSize.snapped(Double(customWidth)), height: PixelSize.snapped(Double(customHeight)))
            : aspect.size(base: resolution)
    }

    var trimmedPrompt: String { prompt.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// Seeds for the next batch: consecutive from the fixed seed, or fresh random ones.
    func nextSeeds() -> [Int] {
        let count = max(1, min(Self.maxBatch, batchCount))
        if randomSeed {
            return (0..<count).map { _ in Int.random(in: 0..<1_000_000_000) }
        }
        return (0..<count).map { seed + $0 }
    }

    /// Picks the preset that produced `size`, or switches to a custom size.
    mutating func apply(size: PixelSize) {
        for aspect in AspectRatio.allCases {
            for resolution in Self.resolutions where aspect.size(base: resolution) == size {
                self.aspect = aspect
                self.resolution = resolution
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
    var seed: Int
    var size: PixelSize
    var steps: Int
    var guidance: Double
    var transparentBackground: Bool
    var lowMemory: Bool
}

enum PromptExamples {
    typealias Example = (title: String, prompt: String)

    static func `for`(_ family: ModelFamily) -> [Example] {
        family == .ming ? design : photo
    }

    static let design: [Example] = [
        (
            "Festival poster",
            "A bold Swiss-style poster for a jazz festival titled 'BLUE NOTES 2026', large condensed headline, "
                + "abstract saxophone shapes, deep blue and warm orange palette, dates 'JULY 12–14' at the bottom"
        ),
        (
            "App icon, transparent background",
            "A glossy 3D app icon of a paper plane on a rounded square, soft gradients from teal to violet, "
                + "subtle inner glow, isolated on a transparent background"
        ),
        (
            "Business card",
            "A minimalist business card for a design studio named 'NORTH & FORM', off-white paper, "
                + "black serif logotype, thin geometric line accent, generous whitespace"
        ),
        (
            "App screen",
            "A clean mobile app screen for a meditation app, headline 'Breathe in', circular progress ring, "
                + "pastel lavender palette, rounded cards, modern sans-serif typography"
        ),
        (
            "Sticker",
            "A cute die-cut sticker of a smiling coffee cup with the text 'BUT FIRST, COFFEE', thick white border, "
                + "flat vector illustration, transparent background"
        ),
    ]

    static let photo: [Example] = [
        (
            "Natural-light portrait",
            "Close-up portrait of an elderly fisherman with a weathered face and a knitted cap, soft window light, "
                + "shallow depth of field, 85mm lens, natural skin texture, muted film colors"
        ),
        (
            "Neon sign",
            "A rainy night street in Tokyo, a small ramen shop with a glowing neon sign that reads 'OPEN LATE', "
                + "reflections on wet asphalt, cinematic lighting, 35mm photo"
        ),
        (
            "Product still life",
            "Studio product photo of a minimalist ceramic coffee cup on a travertine block, "
                + "warm morning light, long soft shadows, beige and terracotta palette, high detail"
        ),
        (
            "Landscape",
            "Aerial view of the Dolomites at sunrise, low clouds between jagged peaks, golden light on the rock faces, "
                + "ultra-detailed landscape photography"
        ),
        (
            "Illustration",
            "A cozy isometric illustration of a tiny bookshop with a cat sleeping in the window, "
                + "warm pastel colors, soft shading, detailed and charming"
        ),
    ]
}
