import Foundation
import MLX
import TurboEngineCore

/// `turbo-engine bench-idol <fixture> [--steps N] [--collapse] [--save <file>]`: the Idol denoiser at its real size (8,192 latent, 4,101 picture
/// and 1 camera tokens) on a recording's picture tokens and noise (Engine/Reference/idol_reference.py --record): the
/// load, both branches' picture tokens, then a whole guided run timed step by step, with the peak memory and what 50
/// and 25 steps would take; `--collapse` runs the unconditional pass with the picture's tokens collapsed to two
/// (`IdolDenoiser.collapsed`); `--save` writes the final latent and camera (`latent`, `camera`) for the reference's
/// decoders. A run of the recording's own length is compared with its final latent and camera. The
/// table goes to build/bench/<date>/idol.md (and .json).
enum BenchIdol {
    static func run(arguments: [String]) -> Bool {
        if arguments.first == "--generate" { return generate(arguments: Array(arguments.dropFirst())) }
        guard let path = arguments.first else {
            print("usage: turbo-engine bench-idol <fixture> [--steps N] [--collapse] [--save <file>]")
            return false
        }
        var steps: Int?
        if let index = arguments.firstIndex(of: "--steps"), index + 1 < arguments.count { steps = Int(arguments[index + 1]) }
        do {
            let fixture = URL(fileURLWithPath: path)
            let json = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.appending(path: "fixture.json"))) as? [String: Any] ?? [:]
            let references = try loadArrays(url: fixture.appending(path: "references.safetensors"))
            func reference(_ name: String) throws -> MLXArray {
                guard let array = references[name] else { throw Verify.VerifyError.missingReference(name) }
                return array
            }
            let pack = URL(fileURLWithPath: json["pack"] as? String ?? "")
            let recordedSteps = (json["steps"] as? NSNumber)?.intValue ?? 50
            let steps = steps ?? recordedSteps
            let guidance = (json["guidance"] as? NSNumber)?.floatValue ?? 3
            let rescale = (json["rescale_t"] as? NSNumber)?.doubleValue ?? 3

            Memory.peakMemory = 0
            var clock = Date()
            func lap() -> Double {
                let now = Date()
                defer { clock = now }
                return now.timeIntervalSince(clock)
            }
            let denoiser = try IdolDenoiser.load(folder: pack)
            let load = lap()
            let weights = Double(Memory.activeMemory) / Double(1 << 30)
            print(String(format: "loaded %@ in %.1f s, %.2f GB active", pack.lastPathComponent, load, weights))

            let feature1 = try reference("feature1")
            let feature2 = try reference("feature2")
            let conditional = denoiser.context(feature1: feature1, feature2: feature2)
            let unconditional = denoiser.context(feature1: zeros(like: feature1), feature2: zeros(like: feature2))
            eval(conditional, unconditional)
            let contexts = lap()
            print(String(format: "picture tokens of both branches refined in %.2f s", contexts))

            var sampler = IdolSampler(steps: steps, guidance: guidance, rescale: rescale)
            sampler.collapsesUnconditional = arguments.contains("--collapse")
            print("\(steps) guided steps, the unconditional pass \(sampler.collapsesUnconditional ? "collapsed to 8,195 tokens" : "whole, 12,294 tokens")")
            var seconds: [Double] = []
            _ = lap()
            let result = try sampler.run(
                denoiser, latent: try reference("noise_latent"), camera: try reference("noise_camera"),
                conditional: conditional, unconditional: unconditional,
                progress: { step, total in
                    seconds.append(lap())
                    if step == 1 || step % 10 == 0 || step == total {
                        print(String(format: "  step %d/%d: %.2f s", step, total, seconds.last!))
                    }
                })
            let denoise = seconds.reduce(0, +)
            let steady = Array(seconds.dropFirst())
            let perStep = steady.isEmpty ? denoise : steady.reduce(0, +) / Double(steady.count)
            let median = steady.isEmpty ? denoise : steady.sorted()[steady.count / 2]
            let peak = Double(Memory.peakMemory) / Double(1 << 30)
            let first = seconds.first ?? 0
            func projected(_ n: Int) -> Double { first + perStep * Double(n - 1) }
            print(String(format: "denoise: %.1f s for %d guided steps; first %.2f s, then %.2f s per step (median %.2f, %.2f to %.2f)",
                         denoise, steps, first, perStep, median, steady.min() ?? first, steady.max() ?? first))
            print(String(format: "projected: 50 steps %.0f s, 25 steps %.0f s; peak memory %.2f GB", projected(50), projected(25), peak))

            if let index = arguments.firstIndex(of: "--save"), index + 1 < arguments.count {
                try save(arrays: ["latent": result.latent, "camera": result.camera], url: URL(fileURLWithPath: arguments[index + 1]))
                print("saved the final latent and camera to \(arguments[index + 1])")
            }
            var parity: [String: Double] = [:]
            if steps == recordedSteps {
                for (name, got, want) in [("latent", result.latent, try reference("final_latent")),
                                          ("camera", result.camera, try reference("final_camera"))] {
                    _ = Verify.report("final \(name)", got: got, want: want, counts: false)
                    let cos = Double(VerifyIdol.cosine(got, want))
                    print(String(format: "      cos %.6f", cos))
                    parity["final_\(name)_cosine"] = cos
                }
            }

            let folder = URL(fileURLWithPath: "build/bench/\(Bench.stamp())", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var row: [String: Any] = [
                "pack": pack.path, "steps": steps, "guidance": guidance, "collapsed": sampler.collapsesUnconditional, "load_seconds": load, "context_seconds": contexts,
                "denoise_seconds": denoise, "first_step_seconds": first, "seconds_per_step": perStep, "median_step_seconds": median,
                "projected_50_steps": projected(50), "projected_25_steps": projected(25), "peak_gigabytes": peak,
                "weights_gigabytes": weights, "step_seconds": seconds,
            ]
            row.merge(parity) { _, new in new }
            try JSONSerialization.data(withJSONObject: row, options: [.prettyPrinted, .sortedKeys]).write(to: folder.appending(path: sampler.collapsesUnconditional ? "idol-collapsed.json" : "idol.json"))
            let table = """
            | Idol denoiser | steps | unconditional pass | load | per guided step | denoise | 50 steps | 25 steps | peak |
            |---|---:|---|---:|---:|---:|---:|---:|---:|
            | \(pack.lastPathComponent) | \(steps) | \(sampler.collapsesUnconditional ? "collapsed" : "whole") | \(String(format: "%.1f s", load)) | \(String(format: "%.2f s", perStep)) | \(String(format: "%.0f s", denoise)) | \(String(format: "%.0f s", projected(50))) | \(String(format: "%.0f s", projected(25))) | \(String(format: "%.2f GB", peak)) |
            """
            let name = sampler.collapsesUnconditional ? "idol-collapsed" : "idol"
            try (table + "\n").write(to: folder.appending(path: "\(name).md"), atomically: true, encoding: .utf8)
            print("wrote \(folder.path)/\(name).md")
            return true
        } catch {
            print("bench-idol failed: \(error)")
            return false
        }
    }

    /// `turbo-engine bench-idol --generate <pack> --out <dir> [--low-ram] [--exact <fixture>] <name>=<picture>:<mode>:<steps>[:<gaussians>] …`:
    /// whole objects through the engine (the `generate` command the app sends, events gathered), each GLB written to
    /// <dir>/<name>.glb, with the seconds of each phase and the peak memory, in <dir>/idol-generate.md (and .json).
    /// `--exact` adds one object made from a recording's own noise (its picture, mode and steps), the reference's.
    static func generate(arguments: [String]) -> Bool {
        guard let pack = arguments.first else {
            print("usage: turbo-engine bench-idol --generate <pack> --out <dir> [--low-ram] [--exact <fixture>] <name>=<picture>:<mode>:<steps>[:<gaussians>] …")
            return false
        }
        var out = URL(fileURLWithPath: "build/bench/\(Bench.stamp())", isDirectory: true)
        var lowRam = false
        var exact: URL?
        var jobs: [(name: String, picture: String, mode: String, steps: Int, gaussians: Int)] = []
        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--out": index += 1; out = URL(fileURLWithPath: arguments[index], isDirectory: true)
            case "--low-ram": lowRam = true
            case "--exact": index += 1; exact = URL(fileURLWithPath: arguments[index], isDirectory: true)
            default:
                let parts = argument.split(separator: "=", maxSplits: 1).map(String.init)
                let fields = parts.count == 2 ? parts[1].split(separator: ":").map(String.init) : []
                guard fields.count >= 3, let steps = Int(fields[2]) else { print("bench-idol: cannot read \(argument)"); return false }
                jobs.append((parts[0], fields[0], fields[1], steps, fields.count > 3 ? Int(fields[3]) ?? 262_144 : 262_144))
            }
            index += 1
        }
        do {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let engine = Engine()
            let collector = Bench.Collector()
            Emitter.shared.sink = { event, fields in collector.receive(event, fields) }
            defer { Emitter.shared.sink = nil }
            var rows: [[String: Any]] = []
            func command(id: String, picture: String, mode: String, steps: Int, gaussians: Int, output: URL) throws -> Command {
                let object: [String: Any] = [
                    "cmd": "generate", "id": id,
                    "model": ["family": "idol", "path": pack, "name": "ddalcu/Idol-MLX-Serve-8bit", "variant": "", "low_ram": lowRam],
                    "params": ["prompt": "", "seed": 42, "width": 1920, "height": 1080, "steps": steps, "guidance": 3.0,
                               "flatten_alpha": true, "output": output.path, "upcoming_prompts": [String](), "images": [picture],
                               "num_gaussians": gaussians, "mode": mode],
                ]
                let data = try JSONSerialization.data(withJSONObject: object)
                return try Wire.command(from: String(decoding: data, as: UTF8.self))
            }
            for (number, job) in jobs.enumerated() {
                let output = out.appending(path: "\(job.name).glb")
                print("\(job.name): \(job.picture), \(job.mode), \(job.steps) steps, \(job.gaussians) Gaussians\(lowRam ? ", Save memory" : "")")
                collector.reset()
                let c = try command(id: "bench-\(number)", picture: job.picture, mode: job.mode, steps: job.steps, gaussians: job.gaussians, output: output)
                engine.generate(id: c.id!, spec: c.model!, params: c.params!)
                FileHandle.standardError.write(Data("\r".utf8))
                guard let done = collector.done else {
                    print("  failed: \(collector.failure ?? "no result")")
                    return false
                }
                let timings = done["timings"] as? [String: Double] ?? [:]
                let peak = Double(done["peak_memory"] as? Int ?? 0) / Double(1 << 30)
                let phases = ["load", "segment", "encode", "denoise", "decode", "save"]
                    .compactMap { name in timings[name].map { String(format: "%@ %.1f", name, $0) } }.joined(separator: " · ")
                print(String(format: "  %.1f s · peak %.2f GB · %d discs · ", done["seconds"] as? Double ?? 0, peak, done["gaussians"] as? Int ?? 0) + phases)
                var row: [String: Any] = ["name": job.name, "picture": job.picture, "mode": job.mode, "steps": job.steps,
                                          "gaussians": job.gaussians, "low_ram": lowRam]
                row.merge(done.filter { $0.key != "id" }) { _, new in new }
                rows.append(row)
            }
            if let exact {
                rows.append(try exactRun(fixture: exact, pack: URL(fileURLWithPath: pack), out: out))
            }
            try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys]).write(to: out.appending(path: "idol-generate.json"))
            var table = ["| object | mode | steps | total | load | segment | encode | denoise | decode | save | discs | peak |",
                         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in rows {
                let t = row["timings"] as? [String: Double] ?? [:]
                func s(_ key: String) -> String { t[key].map { String(format: "%.1f s", $0) } ?? "—" }
                table.append("| \(row["name"] ?? "") | \(row["mode"] ?? "") | \(row["steps"] ?? "") | "
                             + String(format: "%.1f s", row["seconds"] as? Double ?? 0)
                             + " | \(s("load")) | \(s("segment")) | \(s("encode")) | \(s("denoise")) | \(s("decode")) | \(s("save")) | "
                             + "\(row["gaussians"] ?? "") | " + String(format: "%.2f GB", Double(row["peak_memory"] as? Int ?? 0) / Double(1 << 30)) + " |")
            }
            try (table.joined(separator: "\n") + "\n").write(to: out.appending(path: "idol-generate.md"), atomically: true, encoding: .utf8)
            print("wrote \(out.path)/idol-generate.md")
            return true
        } catch {
            print("bench-idol failed: \(error)")
            return false
        }
    }

    /// The recording's object made again from its own noise (`IdolModel.noiseOverride`), straight through the family.
    static func exactRun(fixture: URL, pack: URL, out: URL) throws -> [String: Any] {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.appending(path: "fixture.json"))) as? [String: Any] ?? [:]
        let references = try loadArrays(url: fixture.appending(path: "references.safetensors"))
        guard let eps = references["vae_eps"], let latent = references["noise_latent"], let camera = references["noise_camera"],
              let picture = json["image"] as? String
        else { throw Verify.VerifyError.missingReference("vae_eps, noise_latent, noise_camera or the picture") }
        let model = try IdolModel(folder: pack)
        model.noiseOverride = (eps.transposed(0, 2, 3, 1).asType(.bfloat16), latent, camera)
        var request = FamilyRequest(prompt: "", seed: 42, width: 0, height: 0, steps: (json["steps"] as? NSNumber)?.intValue ?? 50,
                                    guidance: (json["guidance"] as? NSNumber)?.doubleValue ?? 3, flattenAlpha: false, imagePaths: [picture])
        request.mode = json["mode"] as? String
        request.gaussians = 262_144
        let name = "exact-" + URL(fileURLWithPath: picture).deletingPathExtension().lastPathComponent
        print("\(name): the recording's noise, \(request.steps) steps")
        let started = Date()
        let mesh = try model.generateMesh(request, output: out.appending(path: "\(name).glb"), phase: { _ in },
                                          progress: { step, total in FileHandle.standardError.write(Data("\r  step \(step) of \(total)   ".utf8)) },
                                          isCancelled: { false })
        FileHandle.standardError.write(Data("\r".utf8))
        let seconds = Date().timeIntervalSince(started)
        print(String(format: "  %.1f s · %d discs", seconds, mesh.discs))
        return ["name": name, "picture": picture, "mode": request.mode ?? "", "steps": request.steps, "seconds": seconds, "gaussians": mesh.discs]
    }
}
