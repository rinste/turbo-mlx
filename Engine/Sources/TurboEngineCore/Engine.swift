import Foundation
import MLX

/// The worker behind the JSON protocol: keeps one model loaded, generates images with phases,
/// progress and timings, and answers `load`, `cancel` and `unload` like `turbo_worker.py`.
public final class Engine {
    public static let version = "0.4"
    public static let mlxSwiftVersion = "0.32.2"
    /// Families this engine implements, as the app names them.
    public static var families: [String] { FamilyLoader.families }

    private let emitter = Emitter.shared
    private var loaded: (key: String, model: FamilyModel)?
    private let defaultCacheLimit = Memory.cacheLimit
    private var marks: [String: Double] = [:]
    private let cancelLock = NSLock()
    private var cancelledIDs: Set<String> = []

    public init() {}

    // MARK: Commands

    public func cancel(id: String) {
        cancelLock.lock()
        cancelledIDs.insert(id)
        cancelLock.unlock()
    }

    private func isCancelled(_ id: String) -> Bool {
        cancelLock.lock()
        defer { cancelLock.unlock() }
        return cancelledIDs.contains(id)
    }

    private func forget(_ id: String) {
        cancelLock.lock()
        cancelledIDs.remove(id)
        cancelLock.unlock()
    }

    public func unload() {
        loaded = nil
        Memory.clearCache()
        emitter.emit("unloaded")
    }

    /// Loads a model ahead of its first image.
    public func load(_ spec: ModelSpec) {
        marks = [:]
        do {
            _ = try ensureModel(spec, jobID: nil)
        } catch {
            emitter.log("[turbo] could not load the model in advance: \(error.localizedDescription)")
            // So that the app stops showing the load as under way.
            emitter.emit("load_failed", ["path": spec.path, "message": error.localizedDescription])
        }
    }

    private func ensureModel(_ spec: ModelSpec, jobID: String?) throws -> FamilyModel {
        if let loaded, loaded.key == spec.key { return loaded.model }
        guard Self.families.contains(spec.family) else {
            throw EngineError.unsupportedFamily(spec.family)
        }
        loaded = nil
        Memory.clearCache()
        emitter.emit("phase", ["id": jobID.map { $0 as Any } ?? NSNull(), "phase": "loading"])
        let started = Date()
        let model = try FamilyLoader.load(spec)
        let seconds = Date().timeIntervalSince(started)
        marks["load", default: 0] += seconds
        loaded = (spec.key, model)
        emitter.log("[turbo] model loaded in \(String(format: "%.1f", seconds))s: \(spec.path) (\(FamilyLoader.describe(model, spec: spec)))")
        emitter.emit("model_loaded", ["path": spec.path, "seconds": (seconds * 10).rounded() / 10])
        return model
    }

    public func generate(id: String, spec: ModelSpec, params: GenerationParams) {
        let started = Date()
        marks = [:]
        // Metal keeps what the job uses resident while it runs, as mlx-lm and the Python worker did:
        // under memory pressure macOS would otherwise page the weights out, and a 10 s step could
        // take 60. The allowance goes back when the job ends, so an idle engine pins nothing.
        let wired = WiredAllowance.begin()
        defer {
            wired?.end()
            forget(id)
            Memory.clearCache()
        }
        do {
            Memory.peakMemory = 0
            if isCancelled(id) { throw GenerationError.cancelled }
            let model = try ensureModel(spec, jobID: id)
            let lowRam = spec.lowRam == true
            model.lowRam = lowRam
            // What mflux's --low-ram does that suits a long-lived process: a small buffer cache
            // (and, inside the families, tiled decoding and a released text encoder).
            Memory.cacheLimit = lowRam ? 1 << 30 : defaultCacheLimit
            if isCancelled(id) { throw GenerationError.cancelled }

            emitter.emit("phase", ["id": id, "phase": "encoding"])
            mark("encode_start")
            // This prompt, then the queued ones, encode now while the text encoder is resident.
            let upcoming = (params.upcomingPrompts ?? []).filter { !$0.isEmpty && $0 != params.prompt }
            for text in [params.prompt] + upcoming.prefix(8) where !model.isCached(text) {
                try model.encode(text)
                if isCancelled(id) { throw GenerationError.cancelled }
            }
            model.promptsEncoded()

            let request = FamilyRequest(
                prompt: params.prompt, seed: params.seed, width: params.width, height: params.height,
                steps: params.steps, guidance: params.guidance, flattenAlpha: params.flattenAlpha ?? false,
                frames: params.frames, fps: params.fps, imagePath: params.image,
                upscale: params.upscale, softness: params.softness,
                halfPrecision: params.precision == "bf16"
            )
            if let video = model as? VideoFamilyModel {
                var request = request
                request.autoDuration = params.autoDuration ?? false
                try generateVideo(video, request: request, id: id, spec: spec, params: params, started: started)
                return
            }
            let phaseChanged: (GenerationPhase) -> Void = { [self] phase in
                switch phase {
                case .denoising:
                    mark("denoise_start")
                    emitter.emit("phase", ["id": id, "phase": "denoising", "total": params.steps])
                case .decoding:
                    mark("denoise_end")
                    emitter.emit("phase", ["id": id, "phase": "decoding"])
                default:
                    break
                }
            }
            let progressed: (Int, Int) -> Void = { [self] step, total in
                emitter.emit("progress", ["id": id, "step": step, "total": total])
            }
            let cancelled: () -> Bool = { [self] in isCancelled(id) }
            let output = URL(fileURLWithPath: params.output)
            let source = Provenance.sourceType(input: params.image.map { URL(fileURLWithPath: $0) })
            let image: GeneratedImage
            if params.preview == true, let previewing = model as? PreviewingFamilyModel {
                // The image as it forms: a small PNG beside the output, written again at each
                // preview for the app to read, and gone once the image is decoded.
                let previewURL = Self.previewURL(for: output)
                defer { try? FileManager.default.removeItem(at: previewURL) }
                image = try previewing.generate(
                    request, phase: phaseChanged, progress: progressed,
                    preview: { [self] preview in writePreview(preview, to: previewURL, id: id, source: source) },
                    isCancelled: cancelled
                )
            } else {
                image = try model.generate(request, phase: phaseChanged, progress: progressed, isCancelled: cancelled)
            }
            mark("decode_end")

            emitter.emit("phase", ["id": id, "phase": "saving"])
            var metadata: [String: Any] = [
                "engine": "turbo-engine \(Self.version)",
                "model": spec.name ?? spec.path,
                "prompt": params.prompt,
                "seed": params.seed,
                "steps": params.steps,
                "guidance": params.guidance,
                "width": image.width,
                "height": image.height,
            ]
            if request.halfPrecision { metadata["precision"] = "bf16" }
            try ImageOutput.writePNG(image.pixels, to: output, source: source, metadata: metadata)
            mark("save_end")

            let timings = self.timings()
            emitter.log("[turbo] " + timings.map { "\($0.key) \(String(format: "%.1f", $0.value))s" }.sorted().joined(separator: " · "))
            emitter.emit("done", [
                "id": id,
                "path": params.output,
                "seed": params.seed,
                "width": image.width,
                "height": image.height,
                "seconds": (Date().timeIntervalSince(started) * 100).rounded() / 100,
                "peak_memory": Memory.peakMemory,
                "timings": timings,
            ])
        } catch GenerationError.cancelled {
            emitter.emit("cancelled", ["id": id])
        } catch {
            emitter.log("[turbo] \(error)")
            emitter.emit("failed", ["id": id, "message": error.localizedDescription])
        }
    }

    /// A clip: the sound decoded first, then the frames written to the MP4 with it as the decoder
    /// produces them, a PNG of the first frame next to it as the poster.
    private func generateVideo(
        _ model: VideoFamilyModel, request: FamilyRequest, id: String, spec: ModelSpec, params: GenerationParams, started: Date
    ) throws {
        let output = URL(fileURLWithPath: params.output)
        var writer: VideoOutput?
        var sound: (samples: MLXArray?, rate: Int) = (nil, 48000)
        // A length left to the model is settled now that the prompt is encoded, and reported with
        // the first step so the app can plan the clip it will get.
        var request = request
        request.frames = try model.resolvedFrames(request)
        request.autoDuration = false
        let totalSteps = model.totalSteps(request)
        // A clip stopped or failed halfway leaves no partial MP4 behind.
        var clip: GeneratedClip?
        defer { if clip == nil { writer?.cancel() } }
        clip = try model.generateVideo(
            request,
            phase: { [self] phase in
                switch phase {
                case .denoising:
                    mark("denoise_start")
                    var fields: [String: Any] = ["id": id, "phase": "denoising", "total": totalSteps]
                    if let frames = request.frames { fields["frames"] = frames }
                    emitter.emit("phase", fields)
                case .decoding:
                    mark("denoise_end")
                    emitter.emit("phase", ["id": id, "phase": "decoding"])
                default:
                    break
                }
            },
            progress: { [self] step, total in
                emitter.emit("progress", ["id": id, "step": step, "total": total])
            },
            isCancelled: { [self] in isCancelled(id) },
            audio: { samples, rate in sound = (samples, rate) },
            frames: { frames in
                if writer == nil {
                    writer = try VideoOutput(url: output, width: frames.shape[2], height: frames.shape[1],
                                             fps: request.fps ?? 24, audio: sound.samples, sampleRate: sound.rate)
                }
                try writer!.append(frames: frames)
            }
        )
        mark("decode_end")

        emitter.emit("phase", ["id": id, "phase": "encoding_video"])
        guard let writer, let clip else { throw VideoOutput.OutputError.cannotWrite("no frames were decoded") }
        let source = Provenance.sourceType(input: params.image.map { URL(fileURLWithPath: $0) })
        do {
            try writer.finish()
            try Provenance.markVideo(at: output, as: source)
        } catch {
            writer.cancel()
            throw error
        }
        let poster = output.deletingPathExtension().appendingPathExtension("png")
        if let first = writer.firstFrame {
            try ImageOutput.writePNG(first, to: poster, source: source, metadata: [
                "engine": "turbo-engine \(Self.version)",
                "model": spec.name ?? spec.path,
                "prompt": params.prompt,
                "seed": params.seed,
            ])
        }
        mark("save_end")

        let timings = self.timings()
        emitter.log("[turbo] " + timings.map { "\($0.key) \(String(format: "%.1f", $0.value))s" }.sorted().joined(separator: " · "))
        emitter.emit("done", [
            "id": id,
            "path": params.output,
            "poster": poster.path,
            "seed": params.seed,
            "width": clip.width,
            "height": clip.height,
            "frames": writer.frameCount,
            "fps": clip.fps,
            "seconds": (Date().timeIntervalSince(started) * 100).rounded() / 100,
            "peak_memory": Memory.peakMemory,
            "timings": timings,
        ])
    }

    // MARK: Previews

    /// Where a generation's previews go: the output's name with `.preview` before its extension.
    public static func previewURL(for output: URL) -> URL {
        output.deletingPathExtension().appendingPathExtension("preview").appendingPathExtension(output.pathExtension)
    }

    /// Writes a preview where the app looks for it, whole or not at all (written beside it, then
    /// renamed over the last one), and tells the app. A preview that cannot be written is only logged.
    private func writePreview(_ preview: Preview, to url: URL, id: String, source: Provenance.SourceType) {
        let staging = url.appendingPathExtension("tmp")
        do {
            eval(preview.pixels)
            try ImageOutput.writePNG(preview.pixels, to: staging, source: source, metadata: ["preview": preview.step])
            guard rename(staging.path, url.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            emitter.emit("preview", ["id": id, "step": preview.step, "path": url.path])
        } catch {
            try? FileManager.default.removeItem(at: staging)
            emitter.log("[turbo] no preview at step \(preview.step): \(error.localizedDescription)")
        }
    }

    // MARK: Timings

    private func mark(_ name: String) {
        marks[name] = Date().timeIntervalSince1970
    }

    private func timings() -> [String: Double] {
        var spans: [String: Double] = [:]
        if let load = marks["load"] { spans["load"] = load }
        for (name, begin, end) in [
            ("encode", "encode_start", "denoise_start"),
            ("denoise", "denoise_start", "denoise_end"),
            ("decode", "denoise_end", "decode_end"),
            ("save", "decode_end", "save_end"),
        ] {
            if let b = marks[begin], let e = marks[end] { spans[name] = e - b }
        }
        return spans.compactMapValues { value in
            let rounded = (value * 100).rounded() / 100
            return rounded > 0 ? rounded : nil
        }
    }
}

public enum EngineError: LocalizedError {
    case unsupportedFamily(String)
    case missingTextEncoder(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedFamily(let family): "The native engine does not run the \(family) family."
        case .missingTextEncoder(let family): "The \(family) model needs the path of its text encoder."
        }
    }
}

/// A job's wired-memory allowance: mlx-swift's ticket, as large as the GPU's recommended working
/// set (what the Python worker set with `mx.set_wired_limit`), started and ended from the
/// engine's synchronous thread. When the last ticket ends, MLX goes back to the limit it had.
struct WiredAllowance {
    private let ticket: WiredMemoryTicket

    static func begin() -> WiredAllowance? {
        guard let size = GPU.maxRecommendedWorkingSetBytes(), size > 0 else { return nil }
        let allowance = WiredAllowance(ticket: WiredMemoryTicket(size: size, policy: WiredMaxPolicy()))
        let ticket = allowance.ticket
        wait { await ticket.start() }
        return allowance
    }

    func end() {
        let ticket = ticket
        Self.wait { await ticket.end() }
    }

    private static func wait(_ work: @escaping @Sendable () async -> Int) {
        let done = DispatchSemaphore(value: 0)
        Task.detached {
            _ = await work()
            done.signal()
        }
        done.wait()
    }
}
