import Foundation

/// One JSON line written by `turbo_worker.py` on stdout. Fields depend on `event`.
nonisolated struct WorkerEvent: Decodable, Sendable {
    let event: String
    var id: String?
    var phase: String?
    var step: Int?
    var total: Int64?
    var bytes: Int64?
    var cached: Int64?
    var path: String?
    var seed: Int?
    var width: Int?
    var height: Int?
    var seconds: Double?
    var peakMemory: Int64?
    var message: String?
    // "ready"
    var mflux: String?
    var mlx: String?
    var python: String?
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

/// Versions and hardware reported by a worker that started successfully.
nonisolated struct BackendInfo: Equatable, Sendable {
    var mflux: String
    var mlx: String
    var python: String
    var device: String
    var memory: Int64
}

/// Commands understood by `turbo_worker.py serve`.
nonisolated enum WorkerCommand: Sendable {
    case generate(jobID: UUID, model: ModelDescriptor, modelPath: String, request: GenerationRequest, output: URL)
    case cancel(jobID: UUID)
    case unload
    case shutdown

    var jsonLine: String {
        let object: [String: Any] = switch self {
        case let .generate(jobID, model, modelPath, request, output):
            [
                "cmd": "generate",
                "id": jobID.uuidString,
                "model": [
                    "family": model.family.rawValue,
                    "path": modelPath,
                    // mflux infers some variants (FLUX.2 Klein 4B/9B) from the model's name.
                    "name": model.id,
                    "variant": model.variant ?? "",
                    "low_ram": request.lowMemory,
                ] as [String: Any],
                "params": [
                    "prompt": request.prompt,
                    "seed": request.seed,
                    "width": request.size.width,
                    "height": request.size.height,
                    "steps": request.steps,
                    "guidance": request.guidance,
                    "flatten_alpha": !request.transparentBackground,
                    "output": output.path,
                ] as [String: Any],
            ]
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
}
