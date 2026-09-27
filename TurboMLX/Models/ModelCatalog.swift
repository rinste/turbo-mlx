import Foundation

/// A family of models served by the same adapter in `turbo_worker.py`.
nonisolated enum ModelFamily: String, Codable, Hashable, Sendable, CaseIterable {
    case ming
    case zImageTurbo = "z-image-turbo"
    case flux2Klein = "flux2-klein"

    var displayName: String {
        switch self {
        case .ming: "Ming-Image"
        case .zImageTurbo: "Z-Image Turbo"
        case .flux2Klein: "FLUX.2 Klein"
        }
    }

    /// Whether the model outputs RGBA, so a transparent background can be kept.
    var producesAlpha: Bool { self == .ming }

    /// Checkpoint sub-folders that must hold complete safetensors shards.
    var components: [String] {
        switch self {
        case .ming: ["mllm", "connector", "mlp", "transformer", "vae"]
        case .zImageTurbo, .flux2Klein: ["transformer", "text_encoder", "vae"]
        }
    }

    var tokenizerFile: String {
        switch self {
        case .ming: "mllm/tokenizer.json"
        case .zImageTurbo, .flux2Klein: "tokenizer/tokenizer.json"
        }
    }

    /// Hugging Face files to download; mirrors each mflux weight definition.
    var downloadPatterns: [String] {
        let weights = components.flatMap { ["\($0)/*.safetensors", "\($0)/*.json"] }
        switch self {
        case .ming: return weights
        case .zImageTurbo: return weights + ["tokenizer/*"]
        case .flux2Klein: return weights + ["tokenizer/**", "added_tokens.json", "chat_template.jinja"]
        }
    }

    var promptTip: String {
        switch self {
        case .ming: "Ming-Image is made for graphic design: posters, cards, interfaces, logos and typography."
        case .zImageTurbo: "Z-Image Turbo excels at photorealism and at text inside the image (English and Chinese)."
        case .flux2Klein: "FLUX.2 Klein takes seconds per image: great for exploring many variations."
        }
    }
}

nonisolated struct ModelDescriptor: Identifiable, Hashable, Codable, Sendable {
    nonisolated enum Source: Hashable, Codable, Sendable {
        case huggingFace(repo: String)
        case local(path: String)
    }

    var name: String
    var detail: String
    var family: ModelFamily
    var source: Source
    var sizeBytes: Int64?
    /// mflux registry entry (e.g. "flux2-klein-9b") when the name alone does not identify it.
    var variant: String?
    var license: String?
    /// Smallest Mac memory the model runs well in, with Save memory on.
    var recommendedMemoryGB: Int?
    var isBuiltIn = false
    var isRecommended = false

    /// A repo id or a folder path.
    var id: String {
        switch source {
        case .huggingFace(let repo): repo
        case .local(let path): path
        }
    }

    var repo: String? {
        if case .huggingFace(let repo) = source { repo } else { nil }
    }

    var webURL: URL? {
        repo.flatMap { URL(string: "https://huggingface.co/\($0)") }
    }

    /// FLUX.2 Klein "base" checkpoints are not distilled: many steps and real CFG.
    private var isKleinBase: Bool {
        family == .flux2Klein && (variant ?? id).lowercased().contains("base")
    }

    var supportsGuidance: Bool {
        switch family {
        case .ming: true
        case .zImageTurbo: false
        case .flux2Klein: isKleinBase
        }
    }

    var defaultSteps: Int {
        switch family {
        case .ming: 12
        case .zImageTurbo: 9
        case .flux2Klein: isKleinBase ? 50 : 4
        }
    }

    var defaultGuidance: Double {
        isKleinBase ? 4 : 1
    }

    var stepRange: ClosedRange<Int> {
        switch family {
        case .ming: 4...40
        case .zImageTurbo: 4...20
        case .flux2Klein: isKleinBase ? 10...60 : 2...12
        }
    }
}

/// The built-in models, named after the memory they need rather than their quantization.
///
/// Peak MLX memory for one 1024 × 1024 image with Save memory on (M1 Max, mflux 0.20): Ming-Image
/// te5 14.7 GB, Z-Image Turbo q4 8.5 GB, FLUX.2 Klein 4B q4 14.1 GB (its VAE cannot decode in
/// tiles, so Save memory does not lower it). Z-Image Turbo q8 adds the 5.1 GB of larger weights,
/// ~13.6 GB (activations are bf16 either way). Each model gets the smallest common Mac size it
/// peaks under 80% of, leaving room for macOS and other apps.
///
/// One entry per family and size: Ming-Image ships as te5 only, since the DiT stage sets its peak
/// at 1024 px and is the same in every conversion (te4 is merely less faithful, te6/te8 match te5
/// with more memory), and FLUX.2 Klein 8-bit would need 24 GB like the 4-bit one. Those can still
/// be added from "Add Model…".
enum ModelCatalog {
    static let mingID = "joeynyc/Ming-Image-0.1-Design-mflux-q8-te5"
    static let lightID = "mflux-community/z-image-turbo-mflux-q4"

    /// Ming-Image where it fits, otherwise the model that runs in 16 GB.
    static var defaultModelID: String {
        ProcessInfo.processInfo.physicalMemory >= 24 << 30 ? mingID : lightID
    }

    static let builtIn: [ModelDescriptor] = [
        ModelDescriptor(
            name: "Ming-Image 0.1 Design · 24 GB RAM",
            detail: "Graphic design and typography with a transparent background: posters, cards, UI, logos.",
            family: .ming,
            source: .huggingFace(repo: mingID),
            sizeBytes: 19_249_150_027,
            license: "MIT",
            recommendedMemoryGB: 24,
            isBuiltIn: true,
            isRecommended: true
        ),
        ModelDescriptor(
            name: "Z-Image Turbo · 16 GB RAM",
            detail: "Photorealistic images with legible text, in 9 steps. The lighter 4-bit version.",
            family: .zImageTurbo,
            source: .huggingFace(repo: lightID),
            sizeBytes: 5_902_983_962,
            license: "Apache 2.0",
            recommendedMemoryGB: 16,
            isBuiltIn: true
        ),
        ModelDescriptor(
            name: "Z-Image Turbo · 24 GB RAM",
            detail: "Photorealistic images with legible text, in 9 steps. The 8-bit version, closer to the original.",
            family: .zImageTurbo,
            source: .huggingFace(repo: "mflux-community/z-image-turbo-mflux-q8"),
            sizeBytes: 10_992_100_817,
            license: "Apache 2.0",
            recommendedMemoryGB: 24,
            isBuiltIn: true
        ),
        ModelDescriptor(
            name: "FLUX.2 Klein 4B · 24 GB RAM",
            detail: "Black Forest Labs' distilled model: an image in 4 steps, the fastest here.",
            family: .flux2Klein,
            source: .huggingFace(repo: "mflux-community/flux2-klein-4b-mflux-q4"),
            sizeBytes: 4_619_699_678,
            variant: "flux2-klein-4b",
            license: "Apache 2.0",
            recommendedMemoryGB: 24,
            isBuiltIn: true
        ),
    ]

    /// FLUX.2 Klein entries of mflux's registry, for custom checkpoints whose name is ambiguous.
    static let kleinVariants: [(key: String, label: String)] = [
        ("flux2-klein-4b", "4B distilled"),
        ("flux2-klein-9b", "9B distilled"),
        ("flux2-klein-base-4b", "4B base"),
        ("flux2-klein-base-9b", "9B base"),
    ]
}
