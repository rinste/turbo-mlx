#if !SEALED
import Foundation

/// What a tool could not do, in words the agent can act on.
nonisolated struct MCPToolError: LocalizedError {
    var message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// The server's tools: what agents can ask of the app.
extension MCPServer {
    /// The longest `get_job` waits before answering with the job as it stands.
    private static let longestWait = 600.0

    static let tools: [[String: Any]] = [
        [
            "name": "list_models",
            "title": "List models",
            "description": "The models Turbo MLX knows: what each makes (image, video, audio or model3d) and whether it is downloaded. Only downloaded models can generate.",
            "inputSchema": ["type": "object", "properties": [String: Any](), "additionalProperties": false],
            "annotations": ["readOnlyHint": true],
        ],
        [
            "name": "generate",
            "title": "Generate",
            "description": "Queues a generation in Turbo MLX and returns its job at once. Follow it with get_job. Jobs run one at a time, after any the user queued.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "model": ["type": "string", "description": "A model's id (or name) from list_models. It must be downloaded."],
                    "prompt": ["type": "string", "description": "What to make. For a song, the style of the music."],
                    "negative_prompt": ["type": "string", "description": "What the image should not have. Only some models read it, and only with guidance above 1."],
                    "width": ["type": "integer", "description": "Pixels. Images and video; the model's default if left out."],
                    "height": ["type": "integer", "description": "Pixels. Images and video; the model's default if left out."],
                    "steps": ["type": "integer", "description": "Denoising steps, kept within the model's range. The model's default if left out."],
                    "guidance": ["type": "number", "description": "Guidance scale, 1 to 7, for the models that take one."],
                    "seed": ["type": "integer", "description": "Random if left out."],
                    "lyrics": ["type": "string", "description": "A song's lyrics, section tags such as [verse] on lines of their own. Left out, the song is instrumental."],
                    "duration": ["type": "number", "description": "Seconds: a clip's length, a song's longest length or a sound's length."],
                ] as [String: Any],
                "required": ["model"],
            ] as [String: Any],
        ],
        [
            "name": "get_job",
            "title": "Get job",
            "description": "How a job is going: queued, running (with its step), done (with the path of its file on this Mac), failed or cancelled. With wait_seconds, answers when the job ends or the time is up, whichever comes first.",
            "inputSchema": [
                "type": "object",
                "properties": [
                    "job_id": ["type": "string"],
                    "wait_seconds": ["type": "number", "description": "How long to wait for the job to end, up to 600. 0 if left out."],
                ] as [String: Any],
                "required": ["job_id"],
            ] as [String: Any],
            "annotations": ["readOnlyHint": true],
        ],
        [
            "name": "cancel_job",
            "title": "Cancel job",
            "description": "Stops a job that is queued or running.",
            "inputSchema": [
                "type": "object",
                "properties": ["job_id": ["type": "string"]],
                "required": ["job_id"],
            ] as [String: Any],
        ],
    ]

    func call(_ tool: String, _ arguments: [String: Any]) async throws -> [String: Any] {
        guard let app else { throw MCPToolError("Turbo MLX is not ready yet.") }
        switch tool {
        case "list_models":
            return ["models": app.models.map { describe($0, in: app) }]
        case "generate":
            return try generate(arguments, in: app)
        case "get_job":
            let id = try jobID(in: arguments)
            let wait = min(max((arguments["wait_seconds"] as? NSNumber)?.doubleValue ?? 0, 0), Self.longestWait)
            let deadline = Date().addingTimeInterval(wait)
            while Date() < deadline, app.pendingJobs.contains(where: { $0.id == id }) {
                try? await Task.sleep(for: .milliseconds(250))
            }
            return try describe(job: id, in: app)
        case "cancel_job":
            let id = try jobID(in: arguments)
            guard let job = app.pendingJobs.first(where: { $0.id == id }) else {
                _ = try describe(job: id, in: app)
                throw MCPToolError("The job has already ended.")
            }
            app.cancel(job)
            return try describe(job: id, in: app)
        default:
            throw MCPToolError("Unknown tool.")
        }
    }

    private func describe(_ model: ModelDescriptor, in app: AppModel) -> [String: Any] {
        var entry: [String: Any] = [
            "id": model.id,
            "name": model.name,
            "makes": model.family.media.rawValue,
            "family": model.family.rawValue,
            "downloaded": app.isInstalled(model),
            "downloading": app.downloads.isDownloading(model),
            "takes_prompt": model.family.takesPrompt,
            "needs_reference_image": model.family.requiresReferenceImage,
            "default_steps": model.defaultSteps,
        ]
        if let memory = model.recommendedMemoryGB { entry["recommended_memory_gb"] = memory }
        return entry
    }

    private func generate(_ arguments: [String: Any], in app: AppModel) throws -> [String: Any] {
        guard let name = arguments["model"] as? String else { throw MCPToolError("Say which model: an id from list_models.") }
        guard let model = app.models.first(where: { $0.id == name })
            ?? app.models.first(where: { $0.name.caseInsensitiveCompare(name) == .orderedSame || $0.shortName.caseInsensitiveCompare(name) == .orderedSame })
        else { throw MCPToolError("There is no model \"\(name)\": list_models has the ids.") }

        // What the controls start from for this model, then what the agent asked.
        var settings = AppModel.initialSettings(for: model)
        settings.blocks = PromptBlock.defaults(subject: arguments["prompt"] as? String ?? "")
        settings.negativePrompt = arguments["negative_prompt"] as? String ?? ""
        let number = { (key: String) in arguments[key] as? NSNumber }
        if number("width") != nil || number("height") != nil {
            // One side alone makes a square. `GenerationSettings.size(for:)` snaps both to what the model takes.
            let width = (number("width") ?? number("height"))!.intValue, height = (number("height") ?? number("width"))!.intValue
            settings.usesCustomSize = true
            settings.customWidth = min(max(width, 256), 2048)
            settings.customHeight = min(max(height, 256), 2048)
        }
        if let steps = number("steps") { settings.steps = steps.intValue }
        if let guidance = number("guidance"), model.supportsGuidance {
            settings.guidance = min(max(guidance.doubleValue, GenerationSettings.guidanceRange.lowerBound), GenerationSettings.guidanceRange.upperBound)
        }
        if let seed = number("seed") {
            settings.randomSeed = false
            settings.seed = abs(seed.intValue)
        }
        settings.lyrics = arguments["lyrics"] as? String ?? ""
        settings.instrumental = settings.songLyrics.isEmpty
        if let duration = number("duration")?.doubleValue {
            let clip = GenerationSettings.videoDurations
            settings.videoSeconds = min(max(duration, clip.lowerBound), clip.upperBound)
            // Songs and sounds are kept within their own lengths when the job is queued.
            settings.songSeconds = duration
            settings.soundSeconds = duration
        }

        switch app.blocker(for: model, with: settings) {
        case nil:
            break
        case .modelNotDownloaded?, .modelDownloading?:
            throw MCPToolError("\(model.name) is not downloaded yet: the user downloads models in Turbo MLX.")
        case .missingReference?:
            throw MCPToolError("\(model.name) works from a picture, which agents cannot give yet: use a model that takes a prompt alone.")
        case let blocker?:
            throw MCPToolError(blocker.hint)
        }
        guard let job = app.enqueue(model, with: settings).first else { throw MCPToolError("Nothing was queued.") }
        jobs[job.id] = job
        return try describe(job: job.id, in: app)
    }

    private func jobID(in arguments: [String: Any]) throws -> UUID {
        guard let id = (arguments["job_id"] as? String).flatMap(UUID.init(uuidString:)) else {
            throw MCPToolError("Give the job_id that generate returned.")
        }
        return id
    }

    /// The job as it stands: in the history once it has its file, in the queue until then, and
    /// after that only in what the server remembers of the jobs it queued.
    private func describe(job id: UUID, in app: AppModel) throws -> [String: Any] {
        var entry: [String: Any] = ["job_id": id.uuidString]
        if let item = app.history.items.first(where: { $0.id == id }) {
            entry["status"] = "done"
            entry["path"] = app.url(for: item).path
            entry["makes"] = (item.media ?? .image).rawValue
            entry["model"] = item.modelID
            entry["seed"] = item.request.seed
            entry["seconds"] = (item.seconds * 10).rounded() / 10
            if item.request.size.width > 0 {
                entry["width"] = item.request.size.width
                entry["height"] = item.request.size.height
            }
        } else if let job = app.pendingJobs.first(where: { $0.id == id }) {
            entry["model"] = job.model.id
            entry["seed"] = job.request.seed
            if job === app.activeJob {
                entry["status"] = "running"
                entry["phase"] = job.statusLabel
                entry["step"] = job.step
                entry["total_steps"] = job.totalSteps
            } else {
                entry["status"] = "queued"
                entry["jobs_ahead"] = app.pendingJobs.firstIndex(where: { $0.id == id }) ?? 0
            }
        } else if let job = jobs[id] {
            entry["model"] = job.model.id
            entry["status"] = job.failure == nil ? "cancelled" : "failed"
            if let failure = job.failure { entry["message"] = failure }
        } else {
            throw MCPToolError("There is no job \(id.uuidString).")
        }
        return entry
    }
}
#endif
