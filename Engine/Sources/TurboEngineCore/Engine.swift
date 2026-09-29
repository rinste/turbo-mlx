import Foundation
import MLX

/// The worker behind the JSON protocol: keeps one model loaded, generates images with phases,
/// progress and timings, and answers `load`, `cancel` and `unload` like `turbo_worker.py`.
public final class Engine {
    public static let version = "0.3"
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
        defer {
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
                upscale: params.upscale, softness: params.softness
            )
            if let video = model as? VideoFamilyModel {
                try generateVideo(video, request: request, id: id, spec: spec, params: params, started: started)
                return
            }
            let image = try model.generate(
                request,
                phase: { [self] phase in
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
                },
                progress: { [self] step, total in
                    emitter.emit("progress", ["id": id, "step": step, "total": total])
                },
                isCancelled: { [self] in isCancelled(id) }
            )
            mark("decode_end")

            emitter.emit("phase", ["id": id, "phase": "saving"])
            let output = URL(fileURLWithPath: params.output)
            let source = Provenance.sourceType(input: params.image.map { URL(fileURLWithPath: $0) })
            try ImageOutput.writePNG(image.pixels, to: output, source: source, metadata: [
                "engine": "turbo-engine \(Self.version)",
                "model": spec.name ?? spec.path,
                "prompt": params.prompt,
                "seed": params.seed,
                "steps": params.steps,
                "guidance": params.guidance,
                "width": image.width,
                "height": image.height,
            ])
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
                    emitter.emit("phase", ["id": id, "phase": "denoising", "total": totalSteps])
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
