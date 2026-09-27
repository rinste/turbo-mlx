import Foundation
import MLX

/// The worker behind the JSON protocol: keeps one model loaded, generates images with phases,
/// progress and timings, and answers `load`, `cancel` and `unload` like `turbo_worker.py`.
public final class Engine {
    public static let version = "0.2"
    public static let mlxSwiftVersion = "0.31.6"
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
                steps: params.steps, guidance: params.guidance, flattenAlpha: params.flattenAlpha ?? false
            )
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
            try ImageOutput.writePNG(image.pixels, to: output, metadata: [
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

    public var errorDescription: String? {
        switch self {
        case .unsupportedFamily(let family): "The native engine does not run the \(family) family."
        }
    }
}
