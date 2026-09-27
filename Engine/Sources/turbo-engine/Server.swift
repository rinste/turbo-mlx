import Foundation
import MLX
import TurboEngineCore

/// Reads commands from stdin on a thread (so a cancel gets through while an image is being
/// generated) and runs them one at a time on the main thread.
final class Server {
    private let engine = Engine()
    private let emitter = Emitter.shared
    private let queue = CommandQueue()

    func run() {
        let reader = Thread { [queue] in
            while let line = readLine(strippingNewline: true) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { continue }
                queue.push(trimmed)
            }
            queue.push("{\"cmd\": \"shutdown\"}") // stdin closed: the app is gone
        }
        reader.name = "stdin"
        reader.start()

        emitter.emit("ready", [
            "engine": "turbo-engine \(Engine.version)",
            "mlx": "mlx-swift \(Engine.mlxSwiftVersion)",
            "device": Self.deviceName(),
            "memory": Int(ProcessInfo.processInfo.physicalMemory),
            "families": Engine.families,
        ])

        while true {
            let line = queue.pop()
            let command: Command
            do {
                command = try Wire.command(from: line)
            } catch {
                emitter.log("[turbo] invalid command: \(line.prefix(200))")
                continue
            }
            switch command.cmd {
            case "shutdown":
                return
            case "generate":
                guard let id = command.id, let model = command.model, let params = command.params else {
                    emitter.log("[turbo] generate needs id, model and params")
                    continue
                }
                engine.generate(id: id, spec: model, params: params)
            case "load":
                if let model = command.model { engine.load(model) }
            case "unload":
                engine.unload()
            case "cancel":
                break // handled when it arrived
            default:
                emitter.log("[turbo] unknown command: \(command.cmd)")
            }
        }
    }

    /// "Apple M1 Max", as mflux reports it.
    static func deviceName() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Apple Silicon" }
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        return String(cString: buffer)
    }

    /// Commands in arrival order; cancels are applied as soon as they are read.
    private final class CommandQueue: @unchecked Sendable {
        private var lines: [String] = []
        private let condition = NSCondition()
        var engine: Engine?

        func push(_ line: String) {
            // A cancel must not wait behind the generation it targets.
            if line.contains("\"cancel\""),
               let command = try? Wire.command(from: line), command.cmd == "cancel", let id = command.id {
                engine?.cancel(id: id)
                return
            }
            condition.lock()
            lines.append(line)
            condition.signal()
            condition.unlock()
        }

        func pop() -> String {
            condition.lock()
            while lines.isEmpty { condition.wait() }
            let line = lines.removeFirst()
            condition.unlock()
            return line
        }
    }

    init() {
        queue.engine = engine
    }
}
