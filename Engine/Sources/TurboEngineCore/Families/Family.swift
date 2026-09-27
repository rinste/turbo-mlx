import Foundation
import MLX

/// One image, as the app asks for it.
public struct FamilyRequest {
    public var prompt: String
    public var seed: Int
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Double
    /// Composite an RGBA result onto white (families that produce alpha).
    public var flattenAlpha: Bool

    public init(prompt: String, seed: Int, width: Int, height: Int, steps: Int, guidance: Double, flattenAlpha: Bool) {
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.flattenAlpha = flattenAlpha
    }
}

/// A model family the native engine runs: the Swift twin of the `FAMILIES` adapters in
/// `turbo_worker.py`. The engine encodes every prompt of a job first (while the text encoder is
/// resident), tells the model the prompts are done, then generates.
public protocol FamilyModel: AnyObject {
    /// The bits the checkpoint was saved with (nil: unquantized).
    var bits: Int? { get }
    /// "Save memory": decode in tiles where the decoder allows it, and for the families whose
    /// text encoder is large (Ming-Image, Qwen-Image), keep it and the transformer out of memory
    /// at the same time. Set before `encode` and `generate`.
    var lowRam: Bool { get set }
    /// The prompt's encodings are in the cache.
    func isCached(_ prompt: String) -> Bool
    /// Encodes a prompt into the cache, loading the text encoder if it had been released.
    func encode(_ prompt: String) throws
    /// Every prompt of the job is encoded: a family that releases its text encoder in low-RAM
    /// mode does so now.
    func promptsEncoded()
    /// The pipeline after the prompt is encoded. `progress` gets each finished step and
    /// `isCancelled` is consulted between steps (and tiles).
    func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage
}

/// Loads the model a spec names, by family.
public enum FamilyLoader {
    /// Families this engine implements, as the app names them.
    public static let families = ["flux2-klein", "z-image-turbo", "qwen-image", "ming"]

    public static func load(_ spec: ModelSpec, loadTokenizer: Bool = true) throws -> FamilyModel {
        let root = URL(fileURLWithPath: spec.path)
        switch spec.family {
        case "flux2-klein":
            let config = KleinConfig.forModel(name: spec.name, variant: spec.variant)
            return try KleinModel(modelPath: root, config: config, loadTokenizer: loadTokenizer)
        case "z-image-turbo":
            return try ZImageModel(modelPath: root, config: .turbo, loadTokenizer: loadTokenizer)
        case "qwen-image":
            return try QwenImageModel(modelPath: root, config: .qwenImage2512, loadTokenizer: loadTokenizer)
        case "ming":
            return try MingModel(modelPath: root, config: .design, loadTokenizer: loadTokenizer)
        default:
            throw EngineError.unsupportedFamily(spec.family)
        }
    }

    /// A name for the log: "flux2-klein-4b, 4-bit".
    public static func describe(_ model: FamilyModel, spec: ModelSpec) -> String {
        let bits = model.bits.map { "\($0)-bit" } ?? "bf16"
        return "\(spec.family), \(bits)"
    }
}
