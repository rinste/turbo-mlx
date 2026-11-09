import Foundation

// The JSON-lines protocol shared with `turbo_worker.py`: commands arrive on stdin, one object per
// line; events go out on stdout. Field names are snake_case on the wire.

/// A command from the app.
public struct Command: Decodable {
    public let cmd: String
    public let id: String?
    public let model: ModelSpec?
    public let params: GenerationParams?
}

/// Which model to run, as the app describes it.
public struct ModelSpec: Decodable, Equatable {
    public let family: String
    public let path: String
    public let name: String?
    public let variant: String?
    public let lowRam: Bool?
    /// Families whose text encoder is a checkpoint of its own (LTX-2: Gemma 3).
    public let textEncoderPath: String?
    /// The bits an unquantized checkpoint is quantized to on loading (FLUX 3: BFL's repository is
    /// bf16); nil, as it is stored.
    public let quantize: Int?

    /// What identifies a loaded model: the same triple `turbo_worker.py` keys its cache by, and
    /// the bits it is quantized to on loading.
    public var key: String { "\(family)|\(path)|\(variant ?? "")" + (quantize.map { "|q\($0)" } ?? "") }
}

/// One image to generate.
public struct GenerationParams: Decodable {
    public let prompt: String
    /// What classifier-free guidance steers away from (Qwen-Image and its editor, FLUX.2 Klein
    /// base). Absent or empty, the family's own.
    public let negativePrompt: String?
    public let seed: Int
    public let width: Int
    public let height: Int
    public let steps: Int
    public let guidance: Double
    public let flattenAlpha: Bool?
    public let output: String
    public let upcomingPrompts: [String]?
    /// Video families: frame count, frame rate, and the image the clip starts from.
    public let frames: Int?
    public let fps: Double?
    /// LTX-2.5: the length predicted from the prompt, at most `frames`.
    public let autoDuration: Bool?
    public let image: String?
    /// LTX: the image the clip ends on, with or without `image` to start from.
    public let lastImage: String?
    /// The pictures an image is edited from, in order, for the families that read several (FLUX.2
    /// Klein, Qwen-Image Edit); taken over `image` when there are any.
    public let images: [String]?
    /// Image to image (Z-Image, Qwen-Image, FLUX.2 Klein): how much of `image` the result keeps, as
    /// mflux's `--image-strength` (0–1). The image then starts from the picture instead of being
    /// edited from it (`ImageToImage`). Absent or 0, the family's own use of the picture, if any.
    public let imageStrength: Double?
    /// Upscalers (SeedVR2): the factor the picture's shorter side is scaled by, and how much it is
    /// softened first (0–1).
    public let upscale: Double?
    public let softness: Double?
    /// "bf16": the transformer's residual stream in 16 bits where the reference keeps float32
    /// (`FamilyRequest.halfPrecision`). Absent, the reference's precision.
    public let precision: String?
    /// Show the image as it forms: a `preview` event, with a small PNG, at some steps.
    public let preview: Bool?
    /// LoRA files to apply to the model's transformer, each with its strength (families that take
    /// them: `LoRAAdaptable`). Absent or empty, the model as it is.
    public let loras: [LoRASpec]?
    /// LTX-2: the full pipeline, guided, with the pack's dev transformer (`FamilyRequest.quality`).
    public let quality: Bool?
    /// Idol: how many Gaussians to make (32,768 to 1,048,576, a multiple of 32), and how the picture was made:
    /// "synthetic" (a render or a generated picture, cropped tight) or "real" (a photo, cropped looser).
    public let numGaussians: Int?
    public let mode: String?
    /// Songs (MiniMax Music 3): the lyrics, with section tags such as "[verse]" on lines of their own ("[instrumental]"
    /// for none), and the longest the song may last in seconds (up to 360; the model may end it sooner). `prompt` is the
    /// description of the music, `steps` the flow's steps (1 to 30). Sound effects (MOSS-SoundEffect): no lyrics, and
    /// `duration` the clip's length (0.1 to 30 s, in tenths); `prompt` describes the sound, `guidance` and
    /// `negative_prompt` steer it, `steps` are the DiT's (1 to 200).
    public let lyrics: String?
    public let duration: Double?

    /// The pictures given, `images` or else `image`.
    public var imagePaths: [String] {
        if let images, !images.isEmpty { return images }
        return image.map { [$0] } ?? []
    }
}

public enum Wire {
    public static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    public static func command(from line: String) throws -> Command {
        try decoder.decode(Command.self, from: Data(line.utf8))
    }
}

/// Writes events to stdout and log lines to stderr, each line flushed at once.
public final class Emitter: @unchecked Sendable {
    public static let shared = Emitter()
    private let lock = NSLock()
    private let out = FileHandle.standardOutput
    private let err = FileHandle.standardError
    /// Set, it gets the events instead of stdout (`turbo-engine bench` collects them there).
    public var sink: ((String, [String: Any]) -> Void)?

    /// `fields` values must be JSON-representable (String, Int, Double, Bool, [String: Any], [Any], NSNull).
    public func emit(_ event: String, _ fields: [String: Any] = [:]) {
        if let sink {
            sink(event, fields)
            return
        }
        var object = fields
        object["event"] = event
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes]) else { return }
        lock.lock()
        out.write(data)
        out.write(Data("\n".utf8))
        lock.unlock()
    }

    public func log(_ message: String) {
        lock.lock()
        err.write(Data((message + "\n").utf8))
        lock.unlock()
    }
}
