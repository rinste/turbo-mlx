import Foundation
import MLX
import TurboEngineCore

/// `turbo-engine bench`: the catalog's models found in the Hugging Face cache, run through the
/// same `Engine` the app talks to, on fixed prompts, seeds and sizes, with the seconds per phase and
/// the peak memory of each in a table (Markdown, and JSON beside it). What re-measures the figures
/// in docs/ after a change, an mlx-swift update above all: an MLX release can change the memory a
/// model needs without changing a pixel.
///
///   turbo-engine bench [--hub <dir>] [--only <text>]... [--sizes 512,1024]
///                      [--save-memory on|off] [--half] [--repeat N] [--out <dir>]
///
/// By default the requests are the rows of the tables in docs/swift-engine-plan.md (Status), so a
/// run compares with them; `--sizes` replaces the text-to-image rows of the image models with
/// squares of those sides at each model's default steps, the sweep `MemoryEstimate.swift` is
/// fitted on; `--save-memory` sets Save memory for every row; `--half` runs the 16-bit stream
/// option on the models that have it; `--repeat` runs each row several times and keeps the median
/// (and the highest peak). A model that is not in the cache is skipped, and said so. The images,
/// the clips and the tables go to build/bench/<date>/, or `--out`.
enum Bench {
    struct Run {
        var label: String
        var width: Int
        var height: Int
        var steps: Int
        var guidance: Double = 1
        var lowRam = false
        var frames: Int? = nil
        var fps: Double? = nil
        /// The size of the picture an edit starts from or an upscale enlarges, drawn for the run.
        var picture: (width: Int, height: Int)? = nil
        var upscale: Double? = nil
        var prompt = Bench.prompt
    }

    struct Entry {
        var repo: String
        var family: String
        var variant: String? = nil
        /// A second checkpoint the family reads (LTX-2.3's Gemma 3).
        var companion: String? = nil
        var defaultSteps: Int
        var defaultGuidance: Double = 1
        var runs: [Run]
        /// Whether `--sizes` replaces the runs (text to image) or they stay (edits, upscales, clips).
        var sweeps = true
        /// Whether the model has the 16-bit stream option (`--half`).
        var halfPrecision = false
    }

    static let prompt = "A lighthouse on a rocky shore at dusk, warm light in its windows, gulls over the water, painted in oils"
    static let editPrompt = "Make it winter, with snow on the rocks and a grey sky"
    static let clipPrompt = "Waves roll onto a rocky shore at dusk while a lighthouse beam sweeps across the water; gulls call in the wind"
    static let seed = 42

    /// The rows of the Status tables in docs/swift-engine-plan.md, and the README's figures.
    static let suite: [Entry] = [
        Entry(repo: "mflux-community/flux2-klein-4b-mflux-q4", family: "flux2-klein", variant: "flux2-klein-4b", defaultSteps: 4, runs: [
            Run(label: "512², 4 steps", width: 512, height: 512, steps: 4),
            Run(label: "1024², 4 steps", width: 1024, height: 1024, steps: 4),
            Run(label: "edit, 640 × 512, 4 steps", width: 640, height: 512, steps: 4, picture: (640, 512), prompt: editPrompt),
        ]),
        Entry(repo: "mflux-community/z-image-turbo-mflux-q4", family: "z-image-turbo", defaultSteps: 9, runs: [
            Run(label: "512², 9 steps", width: 512, height: 512, steps: 9),
            Run(label: "1024², 9 steps", width: 1024, height: 1024, steps: 9, lowRam: true),
        ], halfPrecision: true),
        Entry(repo: "mflux-community/z-image-turbo-mflux-q8", family: "z-image-turbo", defaultSteps: 9, runs: [
            Run(label: "512², 9 steps", width: 512, height: 512, steps: 9),
        ], halfPrecision: true),
        Entry(repo: "mflux-community/qwen-image-2512-mflux-q4", family: "qwen-image", defaultSteps: 20, defaultGuidance: 4, runs: [
            Run(label: "512², 10 steps, CFG", width: 512, height: 512, steps: 10, guidance: 4, lowRam: true),
        ], halfPrecision: true),
        Entry(repo: "mflux-community/qwen-image-edit-2511-mflux-q4", family: "qwen-image-edit", defaultSteps: 20, defaultGuidance: 4, runs: [
            Run(label: "edit, 320 × 256, 3 steps, CFG", width: 320, height: 256, steps: 3, guidance: 4, lowRam: true, picture: (320, 256), prompt: editPrompt),
        ], sweeps: false, halfPrecision: true),
        Entry(repo: "joeynyc/Ming-Image-0.1-Design-mflux-q8-te5", family: "ming", defaultSteps: 12, runs: [
            Run(label: "512², 12 steps", width: 512, height: 512, steps: 12, lowRam: true),
        ]),
        Entry(repo: "mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit", family: "sensenova", defaultSteps: 8, runs: [
            Run(label: "512², 8 steps", width: 512, height: 512, steps: 8),
            Run(label: "1024², 8 steps", width: 1024, height: 1024, steps: 8),
            Run(label: "edit, 1024², 8 steps", width: 1024, height: 1024, steps: 8, picture: (1024, 1024), prompt: editPrompt),
        ]),
        Entry(repo: "numz/SeedVR2_comfyUI", family: "seedvr2", defaultSteps: 1, runs: [
            Run(label: "640 × 512 → 2×", width: 1280, height: 1024, steps: 1, picture: (640, 512), upscale: 2, prompt: ""),
            Run(label: "768² → 4×", width: 3072, height: 3072, steps: 1, picture: (768, 768), upscale: 4, prompt: ""),
        ], sweeps: false),
        Entry(repo: "dgrauet/ltx-2.3-mlx-q4", family: "ltx-2", companion: "mlx-community/gemma-3-12b-it-4bit", defaultSteps: 8, runs: [
            Run(label: "768 × 512, 25 frames", width: 768, height: 512, steps: 8, frames: 25, fps: 24, prompt: clipPrompt),
            Run(label: "768 × 512, 121 frames", width: 768, height: 512, steps: 8, lowRam: true, frames: 121, fps: 24, prompt: clipPrompt),
        ], sweeps: false),
        Entry(repo: "dgrauet/ltx-2.3-mlx-q8", family: "ltx-2", companion: "mlx-community/gemma-3-12b-it-4bit", defaultSteps: 8, runs: [
            Run(label: "768 × 512, 121 frames", width: 768, height: 512, steps: 8, frames: 121, fps: 24, prompt: clipPrompt),
        ], sweeps: false),
        Entry(repo: "dgrauet/ltx-2.5-mlx-q4", family: "ltx-2.5", defaultSteps: 8, runs: [
            Run(label: "768 × 512, 121 frames", width: 768, height: 512, steps: 8, lowRam: true, frames: 121, fps: 24, prompt: clipPrompt),
        ], sweeps: false),
    ]

    struct BenchError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    struct Options {
        var hub = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub", directoryHint: .isDirectory)
        var only: [String] = []
        var sizes: [Int]?
        var saveMemory: Bool?
        var half = false
        var repeats = 1
        var out: URL?

        init(_ arguments: [String]) throws {
            var iterator = arguments.makeIterator()
            while let argument = iterator.next() {
                func value() throws -> String {
                    guard let next = iterator.next() else { throw BenchError("\(argument) needs a value") }
                    return next
                }
                switch argument {
                case "--hub":
                    hub = URL(fileURLWithPath: (try value() as NSString).expandingTildeInPath, isDirectory: true)
                case "--only":
                    only.append(try value().lowercased())
                case "--sizes":
                    let sides = try value().split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                    guard !sides.isEmpty else { throw BenchError("--sizes takes sides in pixels, e.g. 512,1024") }
                    sizes = sides
                case "--save-memory":
                    switch try value() {
                    case "on": saveMemory = true
                    case "off": saveMemory = false
                    default: throw BenchError("--save-memory takes on or off")
                    }
                case "--half":
                    half = true
                case "--repeat":
                    repeats = max(1, Int(try value()) ?? 1)
                case "--out":
                    out = URL(fileURLWithPath: (try value() as NSString).expandingTildeInPath, isDirectory: true)
                default:
                    throw BenchError("unknown option \(argument)")
                }
            }
        }
    }

    /// What one row measured.
    struct Result {
        var model: String
        var request: String
        var seconds: Double?
        var timings: [String: Double]
        var peakGigabytes: Double?
        var failure: String?
    }

    /// The events of a generation, gathered instead of written to stdout.
    final class Collector {
        var done: [String: Any]?
        var failure: String?

        func reset() {
            done = nil
            failure = nil
        }

        func receive(_ event: String, _ fields: [String: Any]) {
            switch event {
            case "done":
                done = fields
            case "failed":
                failure = fields["message"] as? String ?? "failed"
            case "cancelled":
                failure = "cancelled"
            case "progress":
                if let step = fields["step"] as? Int, let total = fields["total"] as? Int {
                    FileHandle.standardError.write(Data("\r  step \(step) of \(total)   ".utf8))
                }
            case "phase":
                if let phase = fields["phase"] as? String {
                    FileHandle.standardError.write(Data("\r  \(phase)…            ".utf8))
                }
            default:
                break
            }
        }
    }

    static func run(arguments: [String]) -> Bool {
        do {
            let options = try Options(arguments)
            let out = options.out ?? URL(fileURLWithPath: "build/bench/\(stamp())", isDirectory: true)
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let engine = Engine()
            let collector = Collector()
            Emitter.shared.sink = { event, fields in collector.receive(event, fields) }
            defer { Emitter.shared.sink = nil }
            print("bench: results in \(out.path)")

            var results: [Result] = []
            for entry in suite where matches(entry, only: options.only) {
                guard let path = snapshot(of: entry.repo, hub: options.hub) else {
                    print("— \(entry.repo): not in \(options.hub.path), skipped")
                    continue
                }
                var companion: URL?
                if let repo = entry.companion {
                    guard let found = snapshot(of: repo, hub: options.hub) else {
                        print("— \(entry.repo): its text encoder \(repo) is not in the cache, skipped")
                        continue
                    }
                    companion = found
                }
                let half = options.half && entry.halfPrecision
                for run in runs(of: entry, options: options) {
                    let request = run.label + (run.lowRam ? ", Save memory" : "") + (half ? " · 16-bit" : "")
                    print("\(entry.repo): \(request)")
                    var attempts: [Result] = []
                    for repetition in 0 ..< options.repeats {
                        let output = out.appending(path: fileName(entry: entry, run: run, half: half, repetition: repetition))
                        let pictureURL = try run.picture.map { try picture(width: $0.width, height: $0.height, in: out) }
                        collector.reset()
                        try generate(engine, entry: entry, path: path, companion: companion, run: run, half: half, output: output, picture: pictureURL)
                        FileHandle.standardError.write(Data("\r".utf8))
                        var result = Result(model: shortName(entry.repo), request: request, timings: [:])
                        if let done = collector.done {
                            result.seconds = done["seconds"] as? Double
                            result.timings = done["timings"] as? [String: Double] ?? [:]
                            result.peakGigabytes = (done["peak_memory"] as? Int).map { Double($0) / Double(1 << 30) }
                            print("  \(format(result.seconds)) s · peak \(format(result.peakGigabytes)) GB · " + phases(result.timings))
                        } else {
                            result.failure = collector.failure ?? "no result"
                            print("  failed: \(result.failure!)")
                        }
                        attempts.append(result)
                    }
                    results.append(summary(of: attempts))
                }
                engine.unload()
            }

            let table = markdown(results)
            print("")
            print(table)
            try table.write(to: out.appending(path: "bench.md"), atomically: true, encoding: .utf8)
            try json(results).write(to: out.appending(path: "bench.json"), options: .atomic)
            return !results.isEmpty && results.allSatisfy { $0.failure == nil }
        } catch {
            print("bench failed: \(error.localizedDescription)")
            return false
        }
    }

    // MARK: The suite's rows

    static func matches(_ entry: Entry, only: [String]) -> Bool {
        only.isEmpty || only.contains { entry.repo.lowercased().contains($0) || entry.family.contains($0) }
    }

    static func runs(of entry: Entry, options: Options) -> [Run] {
        var runs = entry.runs
        if let sizes = options.sizes, entry.sweeps {
            runs = sizes.map { side in
                Run(label: "\(side)², \(entry.defaultSteps) steps" + (entry.defaultGuidance > 1 ? ", CFG" : ""),
                    width: side, height: side, steps: entry.defaultSteps, guidance: entry.defaultGuidance)
            }
        }
        if let saveMemory = options.saveMemory {
            runs = runs.map { run in
                var run = run
                run.lowRam = saveMemory
                return run
            }
        }
        return runs
    }

    /// The same `generate` line the app sends, through the wire's decoder.
    static func generate(_ engine: Engine, entry: Entry, path: URL, companion: URL?, run: Run, half: Bool, output: URL, picture: URL?) throws {
        var model: [String: Any] = [
            "family": entry.family, "path": path.path, "name": entry.repo, "variant": entry.variant ?? "", "low_ram": run.lowRam,
        ]
        if let companion { model["text_encoder_path"] = companion.path }
        var params: [String: Any] = [
            "prompt": run.prompt, "seed": seed, "width": run.width, "height": run.height, "steps": run.steps,
            "guidance": run.guidance, "flatten_alpha": true, "output": output.path, "upcoming_prompts": [String](),
        ]
        if let frames = run.frames { params["frames"] = frames }
        if let fps = run.fps { params["fps"] = fps }
        if let picture { params["image"] = picture.path }
        if let upscale = run.upscale { params["upscale"] = upscale }
        if half { params["precision"] = "bf16" }
        let object: [String: Any] = ["cmd": "generate", "id": UUID().uuidString, "model": model, "params": params]
        let line = String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        let command = try Wire.command(from: line)
        guard let id = command.id, let spec = command.model, let request = command.params else { throw BenchError("a generate line without its parts") }
        engine.generate(id: id, spec: spec, params: request)
    }

    /// A picture for the runs that start from one: a gradient, a disc and stripes, the same every
    /// time, written once per size.
    static func picture(width: Int, height: Int, in folder: URL) throws -> URL {
        let url = folder.appending(path: "picture-\(width)x\(height).png")
        if FileManager.default.fileExists(atPath: url.path) { return url }
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            let v = Float(y) / Float(max(height - 1, 1))
            for x in 0 ..< width {
                let u = Float(x) / Float(max(width - 1, 1))
                let inDisc = (u - 0.62) * (u - 0.62) + (v - 0.4) * (v - 0.4) < 0.03
                let rgb: (Float, Float, Float) = inDisc
                    ? (0.95, 0.85, 0.3)
                    : (0.15 + 0.7 * u, 0.2 + 0.6 * v, 0.5 + 0.35 * sin(u * 25) * cos(v * 17))
                let offset = (y * width + x) * 3
                bytes[offset] = UInt8((rgb.0 * 255).rounded())
                bytes[offset + 1] = UInt8((rgb.1 * 255).rounded())
                bytes[offset + 2] = UInt8((rgb.2 * 255).rounded())
            }
        }
        try ImageOutput.writePNG(MLXArray(bytes, [height, width, 3]), to: url, source: .trainedAlgorithmicMedia, metadata: ["bench": "picture"])
        return url
    }

    // MARK: The hub cache

    /// The snapshot folder of a repository in the cache: the commit `refs/main` names, else the
    /// snapshot changed last.
    static func snapshot(of repo: String, hub: URL) -> URL? {
        let folder = hub.appending(path: "models--" + repo.replacingOccurrences(of: "/", with: "--"), directoryHint: .isDirectory)
        let snapshots = folder.appending(path: "snapshots", directoryHint: .isDirectory)
        if let ref = try? String(contentsOf: folder.appending(path: "refs/main"), encoding: .utf8) {
            let commit = ref.trimmingCharacters(in: .whitespacesAndNewlines)
            let url = snapshots.appending(path: commit, directoryHint: .isDirectory)
            if !commit.isEmpty, FileManager.default.fileExists(atPath: url.path) { return url }
        }
        let entries = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        func modified(_ url: URL) -> Date {
            (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        return entries.max { modified($0) < modified($1) }
    }

    // MARK: Results

    /// One result for a row run several times: the median run by its seconds, the highest peak.
    static func summary(of attempts: [Result]) -> Result {
        let finished = attempts.filter { $0.seconds != nil }.sorted { $0.seconds! < $1.seconds! }
        guard !finished.isEmpty else { return attempts[0] }
        var result = finished[finished.count / 2]
        result.peakGigabytes = finished.compactMap(\.peakGigabytes).max()
        if finished.count < attempts.count { result.failure = attempts.first { $0.failure != nil }?.failure }
        return result
    }

    static let phaseNames = ["load", "encode", "denoise", "decode", "save"]

    static func markdown(_ results: [Result]) -> String {
        var lines = ["| Model | Request | Total | Load | Encode | Denoise | Decode | Save | Peak |", "|---|---|---|---|---|---|---|---|---|"]
        for result in results {
            if let failure = result.failure, result.seconds == nil {
                lines.append("| \(result.model) | \(result.request) | failed: \(failure) | | | | | | |")
                continue
            }
            let phases = phaseNames.map { format(result.timings[$0]) }.joined(separator: " | ")
            lines.append("| \(result.model) | \(result.request) | \(format(result.seconds)) s | \(phases) | \(format(result.peakGigabytes)) GB |")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    static func json(_ results: [Result]) throws -> Data {
        let rows: [[String: Any]] = results.map { result in
            var row: [String: Any] = ["model": result.model, "request": result.request, "timings": result.timings]
            if let seconds = result.seconds { row["seconds"] = seconds }
            if let peak = result.peakGigabytes { row["peak_gigabytes"] = peak }
            if let failure = result.failure { row["failure"] = failure }
            return row
        }
        let document: [String: Any] = [
            "engine": "turbo-engine \(Engine.version)", "mlx": "mlx-swift \(Engine.mlxSwiftVersion)", "device": Server.deviceName(),
            "date": ISO8601DateFormatter().string(from: Date()), "results": rows,
        ]
        return try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    static func phases(_ timings: [String: Double]) -> String {
        phaseNames.compactMap { name in timings[name].map { "\(name) \(format($0))" } }.joined(separator: " · ")
    }

    static func format(_ value: Double?) -> String {
        value.map { String(format: "%.1f", $0) } ?? "–"
    }

    static func shortName(_ repo: String) -> String {
        repo.split(separator: "/").last.map(String.init) ?? repo
    }

    static func fileName(entry: Entry, run: Run, half: Bool, repetition: Int) -> String {
        let request = run.label.map { $0.isLetter || $0.isNumber ? $0 : "-" }
        let name = "\(shortName(entry.repo))-\(String(request))\(half ? "-bf16" : "")\(repetition > 0 ? "-\(repetition + 1)" : "")"
        return name.replacingOccurrences(of: "--", with: "-") + (run.frames != nil ? ".mp4" : ".png")
    }

    static func stamp() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmm"
        return formatter.string(from: Date())
    }
}
