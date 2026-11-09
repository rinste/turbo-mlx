import Foundation

/// A family of models: one `FamilyModel` in the engine (`Engine/Sources/TurboEngineCore/Families/`).
nonisolated enum ModelFamily: String, Codable, Hashable, Sendable, CaseIterable {
    case ming
    case zImageTurbo = "z-image-turbo"
    case flux2Klein = "flux2-klein"
    case flux3Klein = "flux3-klein"
    case mossImage = "moss-image"
    case qwenImage = "qwen-image"
    case qwenImageEdit = "qwen-image-edit"
    case senseNova = "sensenova"
    case ltx2 = "ltx-2"
    case ltx25 = "ltx-2.5"
    case seedVR2 = "seedvr2"
    case idol
    case minimaxMusic3 = "minimax-music3"
    case mossSoundEffect = "moss-soundeffect"

    var displayName: String {
        switch self {
        case .ming: "Ming-Image"
        case .zImageTurbo: "Z-Image Turbo"
        case .flux2Klein: "FLUX.2 Klein"
        case .flux3Klein: "FLUX 3 Image Klein"
        case .mossImage: "MOSS-Image"
        case .qwenImage: "Qwen-Image"
        case .qwenImageEdit: "Qwen-Image Edit"
        case .senseNova: "SenseNova-U1.5"
        case .ltx2: "LTX-2"
        case .ltx25: "LTX-2.5"
        case .seedVR2: "SeedVR2"
        case .idol: "Idol"
        // The licence asks for this name, as written, wherever a product shows the model.
        case .minimaxMusic3: "MiniMax-Music3"
        case .mossSoundEffect: "MOSS-SoundEffect"
        }
    }

    /// Whether the model outputs RGBA, so a transparent background can be kept.
    var producesAlpha: Bool { self == .ming }

    var media: MediaKind { isLTX ? .video : self == .idol ? .model3D : self == .minimaxMusic3 || self == .mossSoundEffect ? .audio : .image }

    /// Songs with lyrics (MiniMax-Music3), as against sound effects (MOSS-SoundEffect): both are sound.
    var isSong: Bool { self == .minimaxMusic3 }

    /// Lightricks' video models: LTX-2.3 with Gemma 3 beside it, LTX-2.5 with its own Gemma 4.
    var isLTX: Bool { self == .ltx2 || self == .ltx25 }

    /// LTX-2.5 can pick a clip's length from its prompt (its DurationHead).
    var picksDuration: Bool { self == .ltx25 }

    /// The sides of an image are multiples of this: 16 for most (a DiT's 2 × 2 patches of 8-pixel
    /// latents), 32 for SenseNova, whose tokens are 32 × 32 pixels; a clip's, 64 (`videoSize`).
    var sizeMultiple: Int { self == .senseNova ? 32 : 16 }

    /// A reference image: the first frame of a clip (LTX-2), the picture an image is edited from
    /// as the prompt says (FLUX.2 Klein, FLUX 3, Qwen-Image Edit, SenseNova), the picture an upscaler enlarges.
    var takesReferenceImage: Bool {
        isLTX || self == .flux2Klein || self == .flux3Klein || self == .qwenImageEdit || self == .senseNova || isUpscaler || self == .idol
    }

    /// How many reference images the model reads, in order: the engine's `images`, each picture's
    /// tokens after the image's. FLUX 3 takes ten, `Flux3References.maxReferences` (BFL's pipeline);
    /// FLUX.2 Klein has no limit in the engine or mflux, and four is what BFL's specifications give
    /// [klein] 4B and 9B; Qwen-Image Edit has none either, and Qwen's card (from Edit 2509 on, which
    /// first read several) finds it at its best with one to three. The others read one picture
    /// (LTX's last frame is a picture of its own).
    var maxReferenceImages: Int {
        switch self {
        case .flux3Klein: 10
        case .flux2Klein: 4
        case .qwenImageEdit: 3
        default: 1
        }
    }

    /// A second image, the frame a clip ends on (LTX), with or without the first.
    var takesLastFrame: Bool { isLTX }

    /// Qwen-Image Edit only edits, SeedVR2 only upscales and Idol makes an object of the picture: they need it.
    var requiresReferenceImage: Bool { self == .qwenImageEdit || isUpscaler || self == .idol }

    /// A prompt to write: not for an upscaler or Idol, which work from the picture alone.
    var takesPrompt: Bool { !isUpscaler && self != .idol }

    /// Enlarges the reference picture instead of following a prompt: no prompt, no format, a scale.
    var isUpscaler: Bool { self == .seedVR2 }

    /// The families whose reference keeps the transformer's residual stream in float32, which the
    /// "16-bit precision" option trades for speed (the others run in 16 bits already).
    var supportsHalfPrecision: Bool { self == .zImageTurbo || self == .qwenImage || self == .qwenImageEdit }

    /// The families that take LoRAs: those mflux applies them to, with its mappings, and LTX-2 as
    /// ltx-2-mlx does (the engine's `LoRAAdaptable`).
    var takesLoRAs: Bool {
        switch self {
        case .zImageTurbo, .flux2Klein, .flux3Klein, .qwenImage, .qwenImageEdit, .ltx2, .ltx25: true
        case .ming, .mossImage, .senseNova, .seedVR2, .idol, .minimaxMusic3, .mossSoundEffect: false
        }
    }

    /// Files of the model's repository that only the Quality mode needs, downloaded when it is first
    /// chosen: LTX's dev (undistilled) transformer, which its guided first stage runs.
    var qualityFiles: [String] { isLTX ? ["transformer-dev.safetensors"] : [] }

    /// The families that can show the image as it forms: image models, with steps to show it at.
    var showsPreview: Bool { media == .image && !isUpscaler }

    /// Checkpoint sub-folders that must hold complete safetensors shards (SenseNova's MLX packs keep
    /// theirs, and the index naming them, at the top).
    var components: [String] {
        switch self {
        case .ming: ["mllm", "connector", "mlp", "transformer", "vae"]
        case .zImageTurbo, .flux2Klein, .flux3Klein, .mossImage, .qwenImage, .qwenImageEdit: ["transformer", "text_encoder", "vae"]
        case .senseNova, .minimaxMusic3: ["."]
        // The MLX pack: Qwen3's shards as OpenMOSS ships them, the DiT and the VAE converted in `mlx/`.
        case .mossSoundEffect: ["text_encoder"]
        case .ltx2, .ltx25, .seedVR2, .idol: []
        }
    }

    /// Files a complete checkpoint has at its top level (LTX-2 packs keep one file per component;
    /// a `*` stands for a version).
    var requiredFiles: [String] {
        switch self {
        case .ltx2:
            ["embedded_config.json", "transformer-distilled*.safetensors", "connector.safetensors", "vae_decoder.safetensors",
             "vae_encoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors"]
        // LTX-2.5 packs carry the text encoder and its tokenizer, and name the conv VAE `_conv`.
        case .ltx25:
            ["embedded_config.json", "transformer-distilled.safetensors", "connector.safetensors", "text_encoder.safetensors",
             "text_encoder_config.json", "tokenizer.json", "vae_decoder_conv.safetensors", "vae_encoder_conv.safetensors",
             "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_0.safetensors"]
        case .seedVR2: Self.seedVR2Files
        case .senseNova: ["config.json", "model.safetensors.index.json"]
        case .idol: Self.idolFiles
        case .minimaxMusic3: ["config.json", "model.safetensors.index.json"]
        case .mossSoundEffect: ["model_index.json", "transformer/config.json", "mlx/dit.safetensors", "mlx/vae.safetensors"]
        default: []
        }
    }

    var tokenizerFile: String? {
        switch self {
        case .ming: "mllm/tokenizer.json"
        case .zImageTurbo, .flux2Klein, .mossImage, .qwenImage, .qwenImageEdit: "tokenizer/tokenizer.json"
        // Qwen3-VL's, inside the text encoder's folder.
        case .flux3Klein: "text_encoder/tokenizer.json"
        case .senseNova: "tokenizer.json"
        case .ltx2: nil // in the text encoder's checkpoint
        case .ltx25: "tokenizer.json"
        case .seedVR2: nil // no prompt: the engine has the fixed text embedding
        case .idol: nil // no prompt
        case .minimaxMusic3, .mossSoundEffect: "tokenizer/tokenizer.json"
        }
    }

    /// The Idol MLX pack's files: the release's four weight files with BiRefNet among them, DINOv3 and the FLUX.2
    /// VAE's encoder.
    static let idolFiles = ["config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors",
                            "birefnet.safetensors", "dinov3/config.json", "dinov3/model.safetensors", "vae/config.json",
                            "vae/model.safetensors"]

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
        // BFL's repository: the diffusers folders (the text encoder's holds the tokenizer), not the
        // same transformer in BFL's own format at the top, its FP8 and NVFP4 variants or the VAE's
        // second copy.
        case .flux3Klein: return weights + ["LICENSE.md", "NOTICE.md", "README.md"]
        // The diffusers folders (OpenMOSS's custom pipeline code is not read), the scheduler's settings.
        case .mossImage: return weights + ["tokenizer/*", "scheduler/*.json", "model_index.json", "LICENSE", "README.md"]
        case .ltx2:
            return ["LICENSE", "README.md", "config.json", "embedded_config.json", "quantize_config.json", "split_model.json",
                    "transformer-distilled-1.1.safetensors", "connector.safetensors", "vae_decoder.safetensors",
                    "vae_encoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors",
                    "spatial_upscaler_x2_v1_1.safetensors", "spatial_upscaler_x2_v1_1_config.json"]
        // Not the dev transformer, its LoRA, the diffusion decoder (`_av`) or the temporal upscaler.
        case .ltx25:
            return ["LICENSE", "README.md", "embedded_config.json", "quantize_config.json", "split_model.json",
                    "transformer-distilled.safetensors", "connector.safetensors", "text_encoder.safetensors",
                    "text_encoder_config.json", "tokenizer.json", "tokenizer_config.json", "vae_decoder_conv.safetensors",
                    "vae_encoder_conv.safetensors", "audio_vae.safetensors", "vocoder.safetensors",
                    "spatial_upscaler_x2_v1_0.safetensors", "spatial_upscaler_x2_v1_0_config.json"]
        case .seedVR2: return ["README.md"] + Self.seedVR2Files
        case .senseNova: return ["README.md", "config.json", "model.safetensors.index.json", "model-*.safetensors", "tokenizer.json"]
        // With the licences: Idol's (Apache 2.0), its NOTICE, DINOv3's own.
        case .idol: return Self.idolFiles + ["LICENSE", "NOTICE", "README.md", "dinov3/LICENSE.md"]
        // mlx-audio's pack: the weights, their index, the configuration and the tokenizer, with MiniMax's licence; not the
        // scheduler's settings or the chat template, which the engine does not read.
        case .minimaxMusic3:
            return ["LICENSE", "README.md", "config.json", "model.safetensors.index.json", "model-*.safetensors",
                    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
        // The MLX pack's converted DiT and VAE with the settings the engine reads, Qwen3 and its tokenizer.
        case .mossSoundEffect:
            return weights + ["README.md", "model_index.json", "mlx/*.safetensors", "transformer/config.json", "vae/config.json",
                              "scheduler/*.json", "tokenizer/*"]
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
    /// What the Quality mode's files add (`ModelFamily.qualityFiles`).
    var qualitySizeBytes: Int64?
    /// mflux registry entry (e.g. "flux2-klein-9b") when the name alone does not identify it.
    var variant: String?
    /// The bits the engine quantizes an unquantized checkpoint to on loading (FLUX 3: BFL's bf16
    /// repository, as `mflux-generate-flux3 -q 8`); nil, run as stored.
    var quantize: Int?
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

    /// FLUX.2 Klein and FLUX 3 "base" checkpoints are not distilled: many steps and real CFG.
    private var isKleinBase: Bool {
        (family == .flux2Klein || family == .flux3Klein) && (variant ?? id).lowercased().contains("base")
    }

    /// FLUX.2 Klein's transformer width, which its LoRAs must match: 4096 for 9B, 3072 for 4B.
    var kleinWidth: Int? {
        guard family == .flux2Klein else { return nil }
        return (variant ?? id).lowercased().contains("9b") ? 4096 : 3072
    }

    var supportsGuidance: Bool {
        switch family {
        case .ming, .mossImage, .qwenImage, .qwenImageEdit, .idol, .mossSoundEffect: true
        // The catalog's SenseNova packs are the 8-step distillation, which runs without guidance.
        // MiniMax-Music3's guidance (1.5 for its language models, 1.7 for the flow) is the reference's own, fixed.
        case .zImageTurbo, .senseNova, .ltx2, .ltx25, .seedVR2, .minimaxMusic3: false
        case .flux2Klein, .flux3Klein: isKleinBase
        }
    }

    /// The models whose guidance runs against an encoded text, which the user can write: Qwen-Image,
    /// its editor, MOSS-Image and FLUX.2 Klein base. Ming's unconditional pass zeroes the prompt instead, so
    /// there is no text to give it; FLUX 3 Base's is the empty prompt, as BFL's pipeline has it.
    var supportsNegativePrompt: Bool {
        supportsGuidance && family != .ming && family != .flux3Klein && family != .idol
    }

    /// LTX-2: the steps of the first stage (three more refine the upscaled clip).
    var defaultSteps: Int {
        switch family {
        case .ming: 12
        case .zImageTurbo: 9
        case .flux2Klein: isKleinBase ? 50 : 4
        case .flux3Klein: isKleinBase ? 50 : 16
        case .mossImage: 50
        case .qwenImage, .qwenImageEdit: 20
        case .senseNova: 8
        case .ltx2, .ltx25: 8
        case .seedVR2: 1
        case .idol: 50
        // The flow's Euler steps on each window of the song.
        case .minimaxMusic3: 30
        // OpenMOSS's default, on the whole 30 s of latents whatever the length.
        case .mossSoundEffect: 100
        }
    }

    /// Qwen-Image (and its editor) and FLUX.2 Klein base run real CFG, recommended at 4, FLUX 3
    /// Base at mflux's 3.5, MOSS-Image at OpenMOSS's 3; the others default to off.
    var defaultGuidance: Double {
        if family == .flux3Klein { return isKleinBase ? 3.5 : 1 }
        if family == .mossImage || family == .idol { return 3 }
        return family == .qwenImage || family == .qwenImageEdit || family == .mossSoundEffect || isKleinBase ? 4 : 1
    }

    // The Quality mode (`GenerationSettings.fullPipeline(for:)`): LTX's full pipeline runs its dev
    // transformer with guidance for 30 steps, against a negative prompt.

    func stepRange(quality: Bool) -> ClosedRange<Int> { quality && family.isLTX ? 10...50 : stepRange }
    func defaultSteps(quality: Bool) -> Int { quality && family.isLTX ? 30 : defaultSteps }
    func defaultGuidance(quality: Bool) -> Double { quality && family.isLTX ? 3 : defaultGuidance }
    func supportsGuidance(quality: Bool) -> Bool { supportsGuidance || (quality && family.isLTX) }
    func supportsNegativePrompt(quality: Bool) -> Bool { supportsNegativePrompt || (quality && family.isLTX) }

    var stepRange: ClosedRange<Int> {
        switch family {
        case .ming: 4...40
        case .zImageTurbo: 4...20
        case .flux2Klein: isKleinBase ? 10...60 : 2...12
        // A flow map: told each step's target, it holds up with fewer steps than its 16.
        case .flux3Klein: isKleinBase ? 10...60 : 4...32
        case .mossImage: 10...60
        case .qwenImage, .qwenImageEdit: 10...50
        case .senseNova: 4...16
        case .ltx2, .ltx25: 4...8
        case .seedVR2: 1...1
        case .idol: 10...100
        // The reference's own limit is 30.
        case .minimaxMusic3: 8...30
        case .mossSoundEffect: 10...200
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
        // FLUX 3 Image Klein (distilled) from BFL's own repository, whose licence is accepted on
        // Hugging Face: the transformer and Qwen3-VL's language tower in bf16, quantized to 8 bits on
        // loading as mflux's `-q 8` does (its 4 and 6 bits lose too much: 25 and 28 dB from bf16).
        // Measured on an M1 Max: 1024 × 1024 in 140 s, peaking at 11.2 GB with Save memory.
        ModelDescriptor(
            name: "FLUX 3 Image Klein · 16 GB RAM",
            detail: "Black Forest Labs' newest model: photos and legible text in 16 steps; edits a reference image as the prompt says. In 8 bits.",
            family: .flux3Klein,
            source: .huggingFace(repo: "black-forest-labs/flux-3-image-klein"),
            revision: "337337621b8526e120f2275d250242f4a6bedd75",
            sizeBytes: 20_319_306_856,
            variant: "flux3-image-klein",
            quantize: 8,
            license: "FLUX Kommunity",
            recommendedMemoryGB: 16,
            isBuiltIn: true
        ),
        // MOSS-Image SFT (OpenMOSS): the bf16 diffusers checkpoint as mlx-community mirrors it, byte
        // for byte OpenMOSS-Team/MOSS-Image-SFT at fc5de0d5; the transformer and Qwen3.5-9B quantized
        // to 8 bits on loading (30.4 dB from OpenMOSS's bf16 pipeline, 31.6 in bf16). 50 steps of
        // guidance, two passes each, the negative one skipping block 22. Measured on an M1 Max:
        // 1024 × 1024 in 16½ minutes, peaking at 9.1 GB with Save memory (17.1 without).
        ModelDescriptor(
            name: "MOSS-Image · 16 GB RAM",
            detail: "OpenMOSS's 5B model with Qwen3.5 reading the prompt: long text in English and Chinese, layouts from JSON prompts. In 8 bits.",
            family: .mossImage,
            source: .huggingFace(repo: "mlx-community/MOSS-Image-SFT-bf16"),
            revision: "ea698e99312015505884559ebad94b4a5a0d0753",
            sizeBytes: 27_219_787_624,
            quantize: 8,
            license: "Apache 2.0",
            recommendedMemoryGB: 16,
            isBuiltIn: true
        ),
        // SenseNova-U1.5 8B-MoT (SenseTime) with its official 8-step LoRA merged, in 4 bits: the MLX
        // pack mlx-community publishes, made from the original checkpoint for xocialize's Swift
        // runtime. Two 8B stacks (the prompt's and the image's) and a pixel decoder, no VAE.
        // Measured on an M1 Max: 1024 × 1024 in 28 s, peaking at 11.6 GB; an edit of a picture in
        // 34 s (the picture is read with the prompt, about 5 s).
        ModelDescriptor(
            name: "SenseNova-U1.5 · 16 GB RAM",
            detail: "SenseTime's unified model: photos, posters and infographics with legible text, in 8 steps; edits a reference image as the prompt says. The 4-bit version.",
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
            detail: "Lightricks' video model, with sound: a clip from a prompt, or from images as its first and last frames. The 4-bit version.",
            family: .ltx2,
            source: .huggingFace(repo: "dgrauet/ltx-2.3-mlx-q4"),
            revision: "56a5866d638ecfe37c54d348e88938235185c2d4",
            sizeBytes: 20_479_335_378 + 8_068_018_787,
            qualitySizeBytes: 11_322_002_285,
            license: "LTX-2 Community",
            recommendedMemoryGB: 32,
            isBuiltIn: true
        ),
        ModelDescriptor(
            name: "LTX-2.3 · 64 GB RAM",
            detail: "Lightricks' video model, with sound: a clip from a prompt, or from images as its first and last frames. The 8-bit version, closer to the original.",
            family: .ltx2,
            source: .huggingFace(repo: "dgrauet/ltx-2.3-mlx-q8"),
            revision: "6671a7572a530862d1d60ce393b5d93491e3f76b",
            sizeBytes: 29_754_522_642 + 8_068_018_787,
            qualitySizeBytes: 20_597_189_549,
            license: "LTX-2 Community",
            recommendedMemoryGB: 64,
            isBuiltIn: true
        ),
        // LTX-2.5 distilled from dgrauet's 4-bit MLX pack: the same 22B transformer layout, with its
        // own Gemma 4 12B text encoder (4-bit) in the pack, the ancestral sampler in stage 1, and a
        // new conv VAE and upscaler. The repo is gated: its licence is accepted on Hugging Face.
        ModelDescriptor(
            name: "LTX-2.5 · 32 GB RAM",
            detail: "Lightricks' newer video model, with sound: a clip from a prompt, or from images as its first and last frames; follows prompts with several shots. The 4-bit version.",
            family: .ltx25,
            source: .huggingFace(repo: "dgrauet/ltx-2.5-mlx-q4"),
            revision: "e9a20add8c5937fcab635fb8e7a52926487a5776",
            sizeBytes: 28_760_512_454,
            qualitySizeBytes: 11_320_034_990,
            license: "LTX-2 Community",
            recommendedMemoryGB: 32,
            isBuiltIn: true
        ),
        ModelDescriptor(
            name: "Qwen-Image 2512 · 32 GB RAM",
            detail: "Alibaba's 20B model: rich scenes and long text inside the image. Thorough but slow: 20 steps.",
            family: .qwenImage,
            source: .huggingFace(repo: "mflux-community/qwen-image-2512-mflux-q4"),
            revision: "f30c76ff30977000ffdc27fafdd4516bb1c3b44e",
            sizeBytes: 27_605_813_920,
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
            revision: "3c74aab7dd79e6a8148c329cbca78c00c049ec3b",
            sizeBytes: 28_958_984_942,
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
        // Idol (Sperid Labs, Apache 2.0): one picture of one object to 3D Gaussians, written as a GLB of discs. The MLX
        // pack mlx-serve made of the release, its blocks' linears in 8 bits, with BiRefNet, DINOv3 ViT-H+/16 (DINOv3
        // License) and the FLUX.2 VAE's encoder, so nothing gated is fetched. Measured on an M1 Max (1 November 2026):
        // 50 steps in 300 s, 25 in 153 s, 262,144 Gaussians, peaking at 8.3 GB (6.8 with Save memory).
        ModelDescriptor(
            name: "Idol · 16 GB RAM",
            detail: "Sperid Labs' Idol: one picture of one object becomes a 3D object, its sides and back included, as a GLB. No prompt.",
            family: .idol,
            source: .huggingFace(repo: "ddalcu/Idol-MLX-Serve-8bit"),
            revision: "12fbe4b06ef7cc377100b8f60238a6824546cacc",
            sizeBytes: 2_979_218_839,
            license: "Apache 2.0 · DINOv3 License",
            recommendedMemoryGB: 16,
            isBuiltIn: true
        ),
        // MiniMax Music 3 (MiniMax-Music3 Community License): a description and lyrics to a song of up to six minutes,
        // 44.1 kHz stereo. mlx-audio's MLX packs, their language models' and flow transformer's linears in 4 or 8 bits,
        // run as mlx-audio runs them. The 4-bit pack (2 November 2026) peaks at 9.6 GB with Save memory against 14.5 and
        // takes as long; its flow is 17% from the 8-bit pack's float32 after a window's 30 steps, so the same seed gives
        // another song. The licence asks commercial products to show the name "MiniMax-Music3" and every
        // product that lets others generate to keep safeguards against infringing uses (`MusicSection`). Unlike the other
        // families' versions, these two name their bits too, so the picker tells them apart by more than their memory.
        ModelDescriptor(
            name: "MiniMax-Music3 4-bit · 16 GB RAM",
            detail: "MiniMax's music model: a description of the music and its lyrics become a complete song with vocals, up to six minutes long. The lighter 4-bit version.",
            family: .minimaxMusic3,
            source: .huggingFace(repo: "mlx-community/MiniMax-Music3-4bit"),
            revision: "c7ea32923b245fe5afc22d740a1936ad2ac590f3",
            sizeBytes: 9_210_770_326,
            license: "MiniMax-Music3 Community License",
            recommendedMemoryGB: 16,
            isBuiltIn: true
        ),
        ModelDescriptor(
            name: "MiniMax-Music3 8-bit · 24 GB RAM",
            detail: "MiniMax's music model: a description of the music and its lyrics become a complete song with vocals, up to six minutes long. The 8-bit version, closer to the original.",
            family: .minimaxMusic3,
            source: .huggingFace(repo: "mlx-community/MiniMax-Music3-8bit"),
            revision: "10aa4ca578d04c6f5256c1bc22fc8405a09602b5",
            sizeBytes: 14_179_317_410,
            license: "MiniMax-Music3 Community License",
            recommendedMemoryGB: 24,
            isBuiltIn: true
        ),
        // MOSS-SoundEffect v2.0 (OpenMOSS, Apache 2.0): a description to up to 30 s of sound effect or ambience, 48 kHz
        // mono. The MLX pack (its DiT in bf16, the DAC VAE in float32, Qwen3-1.7B as OpenMOSS ships it), run as
        // OpenMOSS's pipeline runs it: every sound denoised as 30 s and cut, 100 guided steps by default.
        ModelDescriptor(
            name: "MOSS-SoundEffect · 16 GB RAM",
            detail: "OpenMOSS's sound-effect model: a description becomes up to 30 seconds of sound, from footsteps and doors to rain, engines and crowds. No music or speech.",
            family: .mossSoundEffect,
            source: .huggingFace(repo: "mlx-community/MOSS-SoundEffect-v2.0-bf16"),
            revision: "71a08a420dcac61b73fc1f721d56ac41078973bf",
            sizeBytes: 8_397_791_730,
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
