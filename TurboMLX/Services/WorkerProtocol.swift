import Foundation

/// One JSON line written by `turbo-engine` on stdout. Fields depend on `event`.
nonisolated struct WorkerEvent: Decodable, Sendable {
    let event: String
    var id: String?
    var phase: String?
    var step: Int?
    var total: Int64?
    var bytes: Int64?
    var cached: Int64?
    var path: String?
    /// "done" for a video: the PNG frame written next to it.
    var poster: String?
    var seed: Int?
    var width: Int?
    var height: Int?
    var seconds: Double?
    var peakMemory: Int64?
    /// A clip's frames: in "done", and in the "denoising" phase once the model picked the length.
    var frames: Int?
    /// Seconds per phase of a finished image: load, encode, denoise, decode, save.
    var timings: [String: Double]?
    var message: String?
    // "ready"
    var engine: String?
    var mlx: String?
    var device: String?
    var memory: Int64?

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    static func parse(_ line: String) -> WorkerEvent? {
        guard line.hasPrefix("{") else { return nil }
        return try? decoder.decode(WorkerEvent.self, from: Data(line.utf8))
    }
}

/// Versions and hardware reported by an engine that started successfully.
nonisolated struct BackendInfo: Equatable, Sendable {
    /// "turbo-engine 0.2".
    var engine: String
    /// "mlx-swift 0.32.2".
    var mlx: String
    var device: String
    var memory: Int64
}

/// Commands understood by `turbo-engine serve`.
nonisolated enum WorkerCommand: Sendable {
    /// `upcomingPrompts` are the distinct prompts (and negative prompts) of the images queued
    /// behind this one: the worker encodes them while the text encoder is resident, so they will
    /// not reload the model.
    /// `textEncoderPath`: the companion checkpoint of a family that has one (LTX-2's Gemma).
    /// `preview`: the engine shows the image as it forms (`preview` events with a small PNG).
    case generate(jobID: UUID, model: ModelDescriptor, modelPath: String, textEncoderPath: String?, request: GenerationRequest,
                  output: URL, upcomingPrompts: [String], preview: Bool)
    /// Loads the model ahead of its first image.
    case load(model: ModelDescriptor, modelPath: String, textEncoderPath: String?, lowMemory: Bool)
    case cancel(jobID: UUID)
    case unload
    case shutdown

    private static func modelObject(_ model: ModelDescriptor, path: String, textEncoderPath: String?, lowMemory: Bool) -> [String: Any] {
        var object: [String: Any] = [
            "family": model.family.rawValue,
            "path": path,
            // The engine infers some variants (FLUX.2 Klein 4B/9B) from the model's name, as mflux does.
            "name": model.id,
            "variant": model.variant ?? "",
            "low_ram": lowMemory,
        ]
        if let textEncoderPath { object["text_encoder_path"] = textEncoderPath }
        return object
    }

    var jsonLine: String {
        let object: [String: Any] = switch self {
        case let .generate(jobID, model, modelPath, textEncoderPath, request, output, upcomingPrompts, preview):
            [
                "cmd": "generate",
                "id": jobID.uuidString,
                "model": Self.modelObject(model, path: modelPath, textEncoderPath: textEncoderPath, lowMemory: request.lowMemory),
                "params": Self.params(request, output: output, upcomingPrompts: upcomingPrompts, preview: preview),
            ]
        case let .load(model, modelPath, textEncoderPath, lowMemory):
            ["cmd": "load", "model": Self.modelObject(model, path: modelPath, textEncoderPath: textEncoderPath, lowMemory: lowMemory)]
        case let .cancel(jobID):
            ["cmd": "cancel", "id": jobID.uuidString]
        case .unload:
            ["cmd": "unload"]
        case .shutdown:
            ["cmd": "shutdown"]
        }
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    private static func params(_ request: GenerationRequest, output: URL, upcomingPrompts: [String], preview: Bool) -> [String: Any] {
        var params: [String: Any] = [
            "prompt": request.prompt,
            "seed": request.seed,
            "width": request.size.width,
            "height": request.size.height,
            "steps": request.steps,
            "guidance": request.guidance,
            "flatten_alpha": !request.transparentBackground,
            "output": output.path,
            "upcoming_prompts": upcomingPrompts,
        ]
        if let negativePrompt = request.negativePrompt { params["negative_prompt"] = negativePrompt }
        if let frames = request.frames { params["frames"] = frames }
        if let fps = request.fps { params["fps"] = Double(fps) }
        if request.autoDuration == true { params["auto_duration"] = true }
        if let upscale = request.upscale { params["upscale"] = upscale }
        if let softness = request.softness { params["softness"] = softness }
        if request.halfPrecision == true { params["precision"] = "bf16" }
        if preview { params["preview"] = true }
        if let loras = request.loras, !loras.isEmpty {
            params["loras"] = loras.map { ["path": LoRALibrary.url($0.file).path, "scale": $0.scale] as [String: Any] }
        }
        if let reference = request.referenceImage {
            let url = HistoryStore.referenceURL(reference)
            if FileManager.default.fileExists(atPath: url.path) { params["image"] = url.path }
        }
        return params
    }
}
