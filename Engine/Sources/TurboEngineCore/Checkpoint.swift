import Foundation
import MLX
import MLXNN

/// A checkpoint saved by mflux: one folder per component (`transformer/`, `text_encoder/`,
/// `vae/`), shards named by `model.safetensors.index.json` (or every `*.safetensors` there), with
/// `quantization_level` and `mflux_version` in each shard's metadata. Tensors are keyed by mflux's
/// module tree, which the Swift modules mirror, so a key maps to a parameter path unchanged.
public struct Checkpoint {
    public enum CheckpointError: LocalizedError {
        case missingComponent(String, URL)
        case missingShard(String, URL)

        public var errorDescription: String? {
            switch self {
            case .missingComponent(let name, let url): "The checkpoint at \(url.path) has no \(name) weights."
            case .missingShard(let name, let url): "The checkpoint at \(url.path) names the shard \(name), which is not on disk."
            }
        }
    }

    public let root: URL
    /// The bits the checkpoint was saved with, from the metadata (nil: unquantized).
    public private(set) var bits: Int?
    public private(set) var mfluxVersion: String?

    public init(root: URL) {
        self.root = root
    }

    /// Every tensor of a component, all shards merged, plus the metadata of the first shard.
    public mutating func loadComponent(_ name: String) throws -> [String: MLXArray] {
        let folder = root.appending(path: name, directoryHint: .isDirectory)
        let shards = try Self.shards(in: folder, component: name)
        var arrays: [String: MLXArray] = [:]
        for (index, shard) in shards.enumerated() {
            let (loaded, metadata) = try loadArraysAndMetadata(url: shard)
            if index == 0 {
                if let level = metadata["quantization_level"], let value = Int(level) { bits = value }
                if let version = metadata["mflux_version"] { mfluxVersion = version }
            }
            arrays.merge(loaded) { _, new in new }
        }
        return arrays
    }

    /// The shards the index names, in its order; without an index, the safetensors of the folder.
    static func shards(in folder: URL, component: String) throws -> [URL] {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: folder.path) else {
            throw CheckpointError.missingComponent(component, folder.deletingLastPathComponent())
        }
        let index = folder.appending(path: "model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let weightMap = json["weight_map"] as? [String: String] {
            let names = Array(Set(weightMap.values)).sorted()
            return try names.map { name in
                let url = folder.appending(path: name)
                guard fileManager.fileExists(atPath: url.path) else { throw CheckpointError.missingShard(name, folder) }
                return url
            }
        }
        let files = (try? fileManager.contentsOfDirectory(atPath: folder.path)) ?? []
        let shards = files.filter { $0.hasSuffix(".safetensors") && !$0.hasPrefix("._") }.sorted()
        guard !shards.isEmpty else { throw CheckpointError.missingComponent(component, folder.deletingLastPathComponent()) }
        return shards.map { folder.appending(path: $0) }
    }
}

public enum WeightLoading {
    public enum LoadError: LocalizedError {
        case update(String)

        public var errorDescription: String? {
            switch self {
            case .update(let message): "The weights do not match the model: \(message)"
            }
        }
    }

    static let inferableBits: Set<Int> = [2, 3, 4, 5, 6, 8]
    static let inferableGroupSizes: Set<Int> = [32, 64, 128]

    /// Puts a component's tensors into a module. Layers the checkpoint stores quantized (a
    /// `scales` tensor next to the packed `weight`) are turned into quantized layers first, with the
    /// bits and group size read off the stored shapes exactly as mflux's loader does, so mixed
    /// precision and any of the supported levels load without being told (linears, embeddings and
    /// the stacked experts of a mixture of experts alike). `ignoring` drops keys the module has no
    /// parameter for (buffers the reference implementation stores, or components this port does
    /// not use).
    public static func apply(
        _ tensors: [String: MLXArray],
        to module: Module,
        ignoring: (String) -> Bool = { _ in false }
    ) throws {
        var weights: [String: MLXArray] = [:]
        for (key, value) in tensors where !ignoring(key) { weights[key] = value }

        do {
            let leaves = module.leafModules().flattened()
            let replacements = leaves.compactMap { (path, layer) -> (String, Module)? in
                guard let (groupSize, bits) = storedQuantization(of: layer, at: path, in: weights),
                      let quantized = quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: .affine)
                else { return nil }
                return (path, quantized)
            }
            if !replacements.isEmpty {
                let layers = Dictionary(leaves, uniquingKeysWith: { first, _ in first })
                let tree = completingLists(NestedItem.unflattened(replacements), at: "", layers: layers)
                try module.update(modules: NestedDictionary(item: tree), verify: .none)
            }
            try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        } catch {
            throw LoadError.update(String(describing: error))
        }
    }

    /// The bits and group size a layer is stored with, read off the packed weight and the scales;
    /// nil for a layer stored as it is.
    static func storedQuantization(of layer: Module, at path: String, in weights: [String: MLXArray]) -> (groupSize: Int, bits: Int)? {
        guard let scales = weights["\(path).scales"], let packed = weights["\(path).weight"] else { return nil }
        let inputDims: Int
        if let linear = layer as? Linear {
            inputDims = linear.weight.shape[1]
        } else if let embedding = layer as? Embedding {
            inputDims = embedding.weight.shape[1]
        } else if let experts = layer as? SwitchLinear {
            // Stacked experts [E, out, in]: the packed width and the scales are per expert row.
            inputDims = experts.weight.shape[2]
        } else {
            return nil
        }
        guard inputDims > 0, let packedWidth = packed.shape.last, let scalesWidth = scales.shape.last, scalesWidth > 0
        else { return nil }
        let bits = packedWidth * 32 / inputDims
        let groupSize = inputDims / scalesWidth
        guard inferableBits.contains(bits), inferableGroupSizes.contains(groupSize) else {
            Emitter.shared.log("[turbo] cannot read the quantization of \(path) (bits \(bits), group \(groupSize))")
            return nil
        }
        return (groupSize: groupSize, bits: bits)
    }

    /// MLXNN replaces the modules of a list only when the update starts with the list's first
    /// element, and keeps only as many as it is given: a list whose first layer stays as it is (the
    /// S3-DiT's `cap_embedder`, an RMSNorm then a quantized linear) or whose first block has
    /// nothing quantized would not load. This fills the gaps: a list of layers gets every layer,
    /// the ones that stay passed as themselves, and a list of blocks an empty update for the blocks
    /// with nothing to replace.
    static func completingLists(_ item: NestedItem<String, Module>, at path: String, layers: [String: Module]) -> NestedItem<String, Module> {
        let prefix = path.isEmpty ? "" : "\(path)."
        switch item {
        case .dictionary(let children):
            var completed: [String: NestedItem<String, Module>] = [:]
            for (key, child) in children {
                completed[key] = completingLists(child, at: prefix + key, layers: layers)
            }
            return .dictionary(completed)
        case .array(let elements) where elements.contains(where: { if case .value = $0 { true } else { false } }):
            var count = elements.count
            while layers["\(prefix)\(count)"] != nil { count += 1 }
            return .array((0 ..< count).map { index in
                if index < elements.count, case .value = elements[index] { return elements[index] }
                return layers["\(prefix)\(index)"].map { .value($0) } ?? .none
            })
        case .array(let elements):
            return .array(elements.enumerated().map { index, element in
                if case .none = element { return .dictionary([:]) }
                return completingLists(element, at: "\(prefix)\(index)", layers: layers)
            })
        default:
            return item
        }
    }
}
