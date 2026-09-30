import Foundation

/// A family of models: one `FamilyModel` in the engine (`Engine/Sources/TurboEngineCore/Families/`).
nonisolated enum ModelFamily: String, Codable, Hashable, Sendable, CaseIterable {
    case ming
    case zImageTurbo = "z-image-turbo"
    case flux2Klein = "flux2-klein"
    case qwenImage = "qwen-image"
    case qwenImageEdit = "qwen-image-edit"
    case senseNova = "sensenova"
    case ltx2 = "ltx-2"
    case seedVR2 = "seedvr2"

    var displayName: String {
        switch self {
        case .ming: "Ming-Image"
        case .zImageTurbo: "Z-Image Turbo"
        case .flux2Klein: "FLUX.2 Klein"
        case .qwenImage: "Qwen-Image"
        case .qwenImageEdit: "Qwen-Image Edit"
        case .senseNova: "SenseNova-U1.5"
        case .ltx2: "LTX-2"
        case .seedVR2: "SeedVR2"
        }
    }

    /// Whether the model outputs RGBA, so a transparent background can be kept.
    var producesAlpha: Bool { self == .ming }

    var media: MediaKind { self == .ltx2 ? .video : .image }

    /// The sides of an image are multiples of this: 16 for most (a DiT's 2 × 2 patches of 8-pixel
    /// latents), 32 for SenseNova, whose tokens are 32 × 32 pixels; a clip's, 64 (`videoSize`).
    var sizeMultiple: Int { self == .senseNova ? 32 : 16 }

    /// A reference image: the first frame of a clip (LTX-2), the picture an image is edited from
    /// as the prompt says (FLUX.2 Klein, Qwen-Image Edit), the picture an upscaler enlarges.
    var takesReferenceImage: Bool { self == .ltx2 || self == .flux2Klein || self == .qwenImageEdit || isUpscaler }

    /// Qwen-Image Edit only edits and SeedVR2 only upscales: they need the picture.
    var requiresReferenceImage: Bool { self == .qwenImageEdit || isUpscaler }

    /// Enlarges the reference picture instead of following a prompt: no prompt, no format, a scale.
    var isUpscaler: Bool { self == .seedVR2 }

    /// Checkpoint sub-folders that must hold complete safetensors shards (SenseNova's MLX packs keep
    /// theirs, and the index naming them, at the top).
    var components: [String] {
        switch self {
        case .ming: ["mllm", "connector", "mlp", "transformer", "vae"]
        case .zImageTurbo, .flux2Klein, .qwenImage, .qwenImageEdit: ["transformer", "text_encoder", "vae"]
        case .senseNova: ["."]
        case .ltx2, .seedVR2: []
        }
    }

    /// Files a complete checkpoint has at its top level (LTX-2 packs keep one file per component;
    /// a `*` stands for a version).
    var requiredFiles: [String] {
        switch self {
        case .ltx2:
            ["embedded_config.json", "transformer-distilled*.safetensors", "connector.safetensors", "vae_decoder.safetensors",
             "vae_encoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors"]
        case .seedVR2: Self.seedVR2Files
        case .senseNova: ["config.json", "model.safetensors.index.json"]
        default: []
        }
    }

    var tokenizerFile: String? {
        switch self {
        case .ming: "mllm/tokenizer.json"
        case .zImageTurbo, .flux2Klein, .qwenImage, .qwenImageEdit: "tokenizer/tokenizer.json"
        case .senseNova: "tokenizer.json"
        case .ltx2: nil // in the text encoder's checkpoint
        case .seedVR2: nil // no prompt: the engine has the fixed text embedding
        }
    }

    /// SeedVR2's original checkpoint (numz/SeedVR2_comfyUI) holds every size and precision; the 3B
    /// model in float16 is these two files.
    static let seedVR2Files = ["seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"]

    /// Hugging Face files to download; mirrors each mflux weight definition (for LTX-2, the files
    /// of dgrauet's packs that the distilled pipeline reads; for SenseNova, the MLX pack's weights,
    /// configuration and tokenizer, not the vocabulary and merges it was made from).
    var downloadPatterns: [String] {
        let weights = components.flatMap { ["\($0)/*.safetensors", "\($0)/*.json"] }
        switch self {
        case .ming: return weights
        case .zImageTurbo: return weights + ["tokenizer/*"]
        case .flux2Klein, .qwenImage, .qwenImageEdit: return weights + ["tokenizer/**", "added_tokens.json", "chat_template.jinja"]
        case .ltx2:
            return ["LICENSE", "README.md", "config.json", "embedded_config.json", "quantize_config.json", "split_model.json",
                    "transformer-distilled-1.1.safetensors", "connector.safetensors", "vae_decoder.safetensors",
                    "vae_encoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors",
                    "spatial_upscaler_x2_v1_1.safetensors", "spatial_upscaler_x2_v1_1_config.json"]
        case .seedVR2: return ["README.md"] + Self.seedVR2Files
        case .senseNova: return ["README.md", "config.json", "model.safetensors.index.json", "model-*.safetensors", "tokenizer.json"]
        }
    }

    /// A second checkpoint the family needs, downloaded with the model and shared by every model
    /// of the family: LTX-2's text encoder, Gemma 3 12B in 4 bits.
    var companion: Companion? {
        switch self {
        // Its model.safetensors.index.json lists the five shards of the bf16 original, not the
        // two it has (mlx-lm globs the folder instead), so the shards are named here.
        case .ltx2: Companion(repo: "mlx-community/gemma-3-12b-it-4bit", revision: "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
                              patterns: ["*.json", "*.safetensors", "tokenizer.model"],
                              requiredFiles: ["config.json", "tokenizer.json", "model-00001-of-00002.safetensors",
                                              "model-00002-of-00002.safetensors"],
                              name: "Gemma 3 12B", license: "Gemma Terms of Use")
        default: nil
        }
    }

    nonisolated struct Companion: Sendable, Hashable {
        let repo: String
        /// The commit the family was checked with, as for the catalog's models.
        let revision: String
        let patterns: [String]
        let requiredFiles: [String]
        let name: String
        let license: String
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
    /// The commit a catalog model was checked with: its download fetches exactly that one, so a
    /// change upstream never reaches the app untested. Nil (a model added by hand) follows `main`.
    var revision: String?
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

    /// The name without the memory the catalog's names end with ("FLUX.2 Klein 4B · 24 GB RAM" →
    /// "FLUX.2 Klein 4B"), for the picker, which shows the memory as a badge of its own.
    var shortName: String {
        guard let suffix = memoryLabel.map({ " · \($0)" }), name.hasSuffix(suffix) else { return name }
        return String(name.dropLast(suffix.count))
    }

    /// "24 GB RAM": the smallest Mac memory the model runs well in.
    var memoryLabel: String? {
        recommendedMemoryGB.map { "\($0) GB RAM" }
    }

    /// FLUX.2 Klein "base" checkpoints are not distilled: many steps and real CFG.
    private var isKleinBase: Bool {
        family == .flux2Klein && (variant ?? id).lowercased().contains("base")
    }

    var supportsGuidance: Bool {
        switch family {
        case .ming, .qwenImage, .qwenImageEdit: true
        // The catalog's SenseNova packs are the 8-step distillation, which runs without guidance.
        case .zImageTurbo, .senseNova, .ltx2, .seedVR2: false
        case .flux2Klein: isKleinBase
        }
    }

    /// LTX-2: the steps of the first stage (three more refine the upscaled clip).
    var defaultSteps: Int {
        switch family {
        case .ming: 12
        case .zImageTurbo: 9
        case .flux2Klein: isKleinBase ? 50 : 4
        case .qwenImage, .qwenImageEdit: 20
        case .senseNova: 8
        case .ltx2: 8
        case .seedVR2: 1
        }
    }

    /// Qwen-Image (and its editor) and FLUX.2 Klein base run real CFG, recommended at 4; the
    /// others default to off.
    var defaultGuidance: Double {
        family == .qwenImage || family == .qwenImageEdit || isKleinBase ? 4 : 1
    }

    var stepRange: ClosedRange<Int> {
        switch family {
        case .ming: 4...40
        case .zImageTurbo: 4...20
        case .flux2Klein: isKleinBase ? 10...60 : 2...12
        case .qwenImage, .qwenImageEdit: 10...50
        case .senseNova: 4...16
        case .ltx2: 4...8
        case .seedVR2: 1...1
        }
    }
}

/// The built-in models, named after the memory they need rather than their quantization.
///
/// Peak MLX memory for one 1024 × 1024 image with Save memory on (M1 Max, mflux 0.20), also for a
/// new prompt in a worker that already generated: Ming-Image te5 14.7 GB, Z-Image Turbo q4 8.5 GB,
/// FLUX.2 Klein 4B q4 14.1 GB (its VAE cannot decode in tiles, so Save memory does not lower it),
/// Qwen-Image 2512 q4 21.3 GB. Z-Image Turbo q8 adds the 5.1 GB of larger weights, ~13.6 GB
/// (activations are bf16 either way). Each model gets the smallest common Mac size it peaks under
/// 80% of, leaving room for macOS and other apps. Without Save memory the peaks are 34.7 GB
/// (Ming-Image) and 43 GB (Qwen-Image), hence Save memory by default below 64 GB.
///
/// One entry per family and size: Ming-Image ships as te5 only, since the DiT stage sets its peak
/// at 1024 px and is the same in every conversion (te4 is merely less faithful, te6/te8 match te5
/// with more memory), and FLUX.2 Klein 8-bit would need 24 GB like the 4-bit one. Those can still
/// be added from "Add Model…".
///
/// Each entry downloads the commit it was checked with (`revision`, and the companion's): move one
/// forward only after generating with the model at the new commit.
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
            revision: "3adb8aaef779f9b3fa4621acebeba97bb4010d42",
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
            revision: "f427e257d8e6ffa03edd4d9ac554a05809da456c",
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
            revision: "4ccff28917346aa9daae49cd3c477cf5260a242c",
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
            revision: "794cd159538149ad9830848508c31f0ea7088e58",
            sizeBytes: 4_619_699_678,
            variant: "flux2-klein-4b",
            license: "Apache 2.0",
            recommendedMemoryGB: 24,
            isBuiltIn: true
        ),
        // SenseNova-U1.5 8B-MoT (SenseTime) with its official 8-step LoRA merged, in 4 bits: the MLX
        // pack mlx-community publishes, made from the original checkpoint for xocialize's Swift
        // runtime. Two 8B stacks (the prompt's and the image's) and a pixel decoder, no VAE.
        // Measured on an M1 Max: 1024 × 1024 in 28 s, peaking at 11.6 GB.
        ModelDescriptor(
            name: "SenseNova-U1.5 · 16 GB RAM",
            detail: "SenseTime's unified model: photos, posters and infographics with legible text, in 8 steps. The 4-bit version.",
            family: .senseNova,
            source: .huggingFace(repo: "mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit"),
            revision: "ff6d0c2dfe21b19891ae4551e11fcc99f6aa82ae",
            sizeBytes: 11_780_443_196,
            license: "Apache 2.0",
            recommendedMemoryGB: 16,
            isBuiltIn: true
        ),
        // LTX-2.3 distilled from dgrauet's MLX packs, the files its two-stage pipeline reads (the
        // transformer, the connector, the video and audio decoders, the image encoder, the ×2
        // upscaler), plus Gemma 3 12B in 4 bits, shared by both. Measured on an M1 Max: a clip of
        // 2 s at 768 × 512 in 97 s, peaking at 15 GB with Save memory (4-bit).
        ModelDescriptor(
            name: "LTX-2.3 · 32 GB RAM",
            detail: "Lightricks' video model, with sound: a clip from a prompt, or from an image as its first frame. The 4-bit version.",
            family: .ltx2,
            source: .huggingFace(repo: "dgrauet/ltx-2.3-mlx-q4"),
            revision: "56a5866d638ecfe37c54d348e88938235185c2d4",
            sizeBytes: 20_479_335_378 + 8_068_018_787,
            license: "LTX-2 Community",
            recommendedMemoryGB: 32,
            isBuiltIn: true
        ),
        ModelDescriptor(
            name: "LTX-2.3 · 64 GB RAM",
            detail: "Lightricks' video model, with sound: a clip from a prompt, or from an image as its first frame. The 8-bit version, closer to the original.",
            family: .ltx2,
            source: .huggingFace(repo: "dgrauet/ltx-2.3-mlx-q8"),
            revision: "6671a7572a530862d1d60ce393b5d93491e3f76b",
            sizeBytes: 29_754_522_642 + 8_068_018_787,
            license: "LTX-2 Community",
            recommendedMemoryGB: 64,
            isBuiltIn: true
        ),
        ModelDescriptor(
            name: "Qwen-Image 2512 · 32 GB RAM",
            detail: "Alibaba's 20B model: rich scenes and long text inside the image. Thorough but slow: 20 steps.",
            family: .qwenImage,
            source: .huggingFace(repo: "mflux-community/qwen-image-2512-mflux-q4"),
            revision: "ec35d366eeb701838007c1720c3d80f2f7fbf9f4",
            sizeBytes: 27_605_801_473,
            license: "Apache 2.0",
            recommendedMemoryGB: 32,
            isBuiltIn: true
        ),
        // Qwen-Image-Edit 2511: the transformer in 4 bits, Qwen2.5-VL in bf16 with its vision
        // tower, which reads the reference picture into the prompt.
        ModelDescriptor(
            name: "Qwen-Image Edit 2511 · 32 GB RAM",
            detail: "Alibaba's 20B editor: changes the reference picture as the prompt says and keeps the rest. Thorough but slow: 20 steps.",
            family: .qwenImageEdit,
            source: .huggingFace(repo: "mflux-community/qwen-image-edit-2511-mflux-q4"),
            revision: "720ad94d982b3dc22f9122ee96af31221d542d47",
            sizeBytes: 28_958_972_495,
            license: "Apache 2.0",
            recommendedMemoryGB: 32,
            isBuiltIn: true
        ),
        // SeedVR2 3B in float16 (ByteDance's one-step diffusion upscaler, as mflux runs it): the
        // transformer and the VAE of the original checkpoint. Measured on an M1 Max: 688 × 384 to
        // 1376 × 768 in 18 s, 672 × 880 to 1344 × 1760 in 34 s, peaking at 10.5 GB.
        ModelDescriptor(
            name: "SeedVR2 Upscaler 3B · 16 GB RAM",
            detail: "ByteDance's upscaler: enlarges a picture two to four times with sharper, faithful detail, in one step. No prompt.",
            family: .seedVR2,
            source: .huggingFace(repo: "numz/SeedVR2_comfyUI"),
            revision: "09ced71023636e9bc8cdf9cdecfb2625d1e691e8",
            sizeBytes: 6_783_018_808 + 501_324_814 + 41_823,
            license: "Apache 2.0",
            recommendedMemoryGB: 16,
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
