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

    /// What identifies a loaded model: the same triple `turbo_worker.py` keys its cache by.
    public var key: String { "\(family)|\(path)|\(variant ?? "")" }
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
