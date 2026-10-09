import Foundation
import MLX
import MLXNN

/// A family that takes LoRAs: mflux applies them to the transformer, with its family's mapping.
public protocol LoRAAdaptable: AnyObject {
    /// The model's LoRAs, which the engine sets to the request's before each image.
    var loras: LoRAAdapters { get }
    /// The module they go on, nil while it is not loaded (a family that releases it puts them back
    /// with `reapply(on:)` when it loads it again).
    var adaptedModule: Module? { get }
}

/// One LoRA file and how strongly it applies (1: as trained).
public struct LoRASpec: Decodable, Equatable, Sendable {
    public let path: String
    public let scale: Double

    public init(path: String, scale: Double) {
        self.path = path
        self.scale = scale
    }

    public var name: String { URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent }
}

/// The LoRAs on a model. Each adapted layer is replaced by a `LoRALinear` that adds
/// `scale · (x·A)·B` to the layer's own output, as mflux's `LoRALinear` (`FusedLoRALinear` for
/// several) computes it, with B scaled by alpha / rank when the file has an alpha; mflux then bakes
/// the sum into the weights by default, re-quantized, which the engine does not: the layers keep
/// their quantized weights, so other LoRAs or another strength only swap the small matrices and the
/// model is never loaded again.
public final class LoRAAdapters {
    public enum LoRAError: LocalizedError {
        case unreadable(String, String)
        case unsupported(String, String)
        case noMatch(String, String, [String])
        case doesNotFit(String, String, Int, Int, [String])

        public var errorDescription: String? {
            switch self {
            case .unreadable(let name, let reason):
                "Could not read the LoRA \(name): \(reason)"
            case .unsupported(let name, let kind):
                "The LoRA \(name) is a \(kind) adapter, which Turbo MLX does not apply yet."
            case .noMatch(let name, let model, let paths):
                "The LoRA \(name) is not for \(model): none of its layers (\(paths.joined(separator: ", "))) is one of the model’s."
            case .doesNotFit(let name, let model, let failed, let total, let examples):
                "The LoRA \(name) does not fit \(model): \(failed) of the \(total) layers it adapts are not the model’s, or not of its size (\(examples.joined(separator: "; ")))."
            }
        }
    }

    public let table: LoRAMapping.Table
    /// What the model carries (on `module`, or as soon as the family loads it), in order.
    public private(set) var specs: [LoRASpec] = []
    /// The module they are on.
    private weak var module: Module?
    /// The layers replaced, by path, as they were.
    private var originals: [String: Linear] = [:]

    public init(table: LoRAMapping.Table) {
        self.table = table
    }

    /// Puts exactly `specs` on `module` (none: the model as it is; a nil module gets them when it is
    /// loaded), replacing what was there. Nothing changes when they are already on, and a file that
    /// does not fit leaves the model without any. Returns a line per file for the log.
    @discardableResult
    public func set(_ specs: [LoRASpec], on module: Module?) throws -> [String] {
        // Already on, or none asked and none on.
        let moved = module != nil && module !== self.module
        guard specs != self.specs || (moved && !specs.isEmpty) else { return [] }
        let hadSome = !self.specs.isEmpty
        restore()
        self.specs = specs
        guard !specs.isEmpty else { return hadSome ? ["[turbo] LoRAs removed"] : [] }
        guard let module else { return [] }
        do {
            return try apply(specs, to: module)
        } catch {
            self.specs = []
            throw error
        }
    }

    /// The same LoRAs on a module the family has just loaded again.
    @discardableResult
    public func reapply(on module: Module) throws -> [String] {
        guard !specs.isEmpty else { return [] }
        return try set(specs, on: module)
    }

    private func apply(_ specs: [LoRASpec], to module: Module) throws -> [String] {
        let mapping = LoRAMapping(table)
        let leaves = module.leafModules().flattened()
        var linears: [String: Linear] = [:]
        for (path, layer) in leaves {
            if let linear = layer as? Linear { linears[path] = linear }
        }
        var deltas: [String: [LoRADelta]] = [:]
        var log: [String] = []
        for spec in specs {
            let (adapters, line) = try Self.adapters(spec, mapping: mapping, linears: linears, modelName: table.rawValue)
            for (path, delta) in adapters { deltas[path, default: []].append(delta) }
            log.append(line)
        }
        eval(deltas.values.flatMap { $0.flatMap { [$0.down, $0.up] } })

        var replacements: [(String, Module)] = []
        var replaced: [String: Linear] = [:]
        for (path, list) in deltas {
            guard let base = linears[path] else { continue }
            replaced[path] = base
            replacements.append((path, LoRALinear(base: base, deltas: list)))
        }
        try Self.update(module, with: replacements, leaves: leaves)
        self.module = module
        originals = replaced
        return log
    }

    /// Puts the layers back as they were.
    private func restore() {
        defer {
            originals = [:]
            module = nil
        }
        guard let module, !originals.isEmpty else { return }
        try? Self.update(module, with: originals.map { ($0.key, $0.value as Module) }, leaves: module.leafModules().flattened())
    }

    private static func update(_ module: Module, with replacements: [(String, Module)], leaves: [(String, Module)]) throws {
        let layers = Dictionary(leaves, uniquingKeysWith: { first, _ in first })
        let tree = WeightLoading.completingLists(NestedItem.unflattened(replacements), at: "", layers: layers)
        try module.update(modules: NestedDictionary(item: tree), verify: .none)
    }

    // MARK: Reading a file

    /// A file's adapters by the path of the layer each goes on, checked against those layers.
    static func adapters(
        _ spec: LoRASpec, mapping: LoRAMapping, linears: [String: Linear], modelName: String
    ) throws -> (adapters: [String: LoRADelta], log: String) {
        let name = URL(fileURLWithPath: spec.path).lastPathComponent
        let arrays: [String: MLXArray]
        do {
            arrays = try loadArrays(url: URL(fileURLWithPath: spec.path))
        } catch {
            throw LoRAError.unreadable(name, error.localizedDescription)
        }
        if arrays.keys.contains(where: { $0.contains("lokr_") }) { throw LoRAError.unsupported(name, "LoKr") }
        if arrays.keys.contains(where: { $0.hasSuffix(".dora_scale") || $0.contains("lora_magnitude_vector") }) {
            throw LoRAError.unsupported(name, "DoRA")
        }

        // The model's own linear paths with underscores, for the Kohya keys no row names.
        let underscoredLinears = Dictionary(linears.keys.map { (LoRAMapping.underscoredForm($0), $0) }, uniquingKeysWith: { a, _ in a })
        struct Parts {
            var down: MLXArray?
            var up: MLXArray?
            var alpha: Float?
            var split = LoRAMapping.Split.none
        }
        var parts: [String: Parts] = [:]
        var unmatched: [String] = []
        for (key, array) in arrays {
            guard let (parsed, matrix) = LoRAMapping.parse(key) else {
                unmatched.append(key)
                continue
            }
            let path = mapping.renamed(parsed)
            var targets = mapping.targets(of: path)
            if targets.isEmpty {
                // A layer named as the model names it.
                let underscored = LoRAMapping.underscoredPrefixes.first { path.hasPrefix($0) }
                let own = underscored.map { underscoredLinears[String(path.dropFirst($0.count))] } ?? (linears[path] != nil ? path : nil)
                if let own { targets = [(own, .none)] }
            }
            guard !targets.isEmpty else {
                unmatched.append(key)
                continue
            }
            for target in targets {
                var part = parts[target.path] ?? Parts()
                switch matrix {
                case .down: part.down = array
                case .up:
                    part.up = array
                    part.split = target.split
                case .alpha: part.alpha = array.asType(.float32).item(Float.self)
                }
                parts[target.path] = part
            }
        }
        guard !parts.isEmpty else {
            let paths = Set(arrays.keys.compactMap { LoRAMapping.parse($0)?.path }).sorted().prefix(3)
            throw LoRAError.noMatch(name, modelName, Array(paths))
        }

        var adapters: [String: LoRADelta] = [:]
        var failures: [String] = []
        for (path, part) in parts.sorted(by: { $0.key < $1.key }) {
            guard let linear = linears[path] else {
                failures.append("no \(path)")
                continue
            }
            guard let down = part.down, var up = part.up, down.ndim == 2, up.ndim == 2 else {
                failures.append("\(path) lacks a matrix")
                continue
            }
            if part.split != .none {
                let third = up.shape[0] / 3
                let index = part.split == .q ? 0 : part.split == .k ? 1 : 2
                up = up[(index * third) ..< ((index + 1) * third)]
            }
            let (outputs, inputs) = linear.shape
            let rank = down.shape[0]
            guard down.shape[1] == inputs, up.shape[0] == outputs, up.shape[1] == rank else {
                failures.append("\(path): \(outputs) × \(inputs), the LoRA's \(up.shape[0]) × \(down.shape[1])")
                continue
            }
            if let alpha = part.alpha { up = up * (alpha / Float(rank)) }
            adapters[path] = LoRADelta(down: down.T, up: up.T, scale: Float(spec.scale))
        }
        if !failures.isEmpty {
            throw LoRAError.doesNotFit(name, modelName, failures.count, parts.count, Array(failures.prefix(3)))
        }
        var line = "[turbo] LoRA \(name) ×\(spec.scale): \(adapters.count) layers"
        if !unmatched.isEmpty {
            line += ", \(unmatched.count) keys left out (\(unmatched.sorted().prefix(3).joined(separator: ", ")))"
        }
        return (adapters, line)
    }
}

/// One adapter on a layer: `down` [in, rank] and `up` [rank, out] as mflux keeps them (the file's
/// matrices transposed, alpha folded into `up`), in the file's precision.
struct LoRADelta {
    let down: MLXArray
    let up: MLXArray
    let scale: Float
}

/// A linear layer with adapters: its own output plus `scale · (x·down)·up` for each. A subclass of
/// `Linear` so it fits the slots the model declares, with the layer it wraps (quantized or not)
/// kept out of the module tree: the model's parameters stay those of the layer.
final class LoRALinear: Linear {
    private final class Wrapped {
        let base: Linear
        let deltas: [LoRADelta]

        init(base: Linear, deltas: [LoRADelta]) {
            self.base = base
            self.deltas = deltas
        }
    }

    private let wrapped: Wrapped

    init(base: Linear, deltas: [LoRADelta]) {
        wrapped = Wrapped(base: base, deltas: deltas)
        super.init(weight: base.weight, bias: base.bias)
        freeze()
    }

    override var shape: (Int, Int) { wrapped.base.shape }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = wrapped.base(x)
        let deltas = wrapped.deltas
        if deltas.count == 1, let delta = deltas.first {
            return output + delta.scale * matmul(matmul(x, delta.down), delta.up)
        }
        // mflux's FusedLoRALinear: the adapters summed, then added.
        var sum = MLXArray.zeros(like: output)
        for delta in deltas {
            sum = sum + delta.scale * matmul(matmul(x, delta.down), delta.up)
        }
        return output + sum
    }
}
