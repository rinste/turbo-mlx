import Foundation

/// Which layer each key of a LoRA file adapts, as mflux's mappings say: their rows are exported to
/// `LoRAMappingTables.swift` by Engine/Fixtures/export_lora_mappings.py, one per spelling of a
/// layer's path once the prefixes and the matrix names are stripped (`parse`). A key is looked up
/// with each number in it standing for the block, as mflux's loader tries them.
public struct LoRAMapping {
    /// The part of a fused projection's up matrix a layer takes: BFL's `qkv` feeds the query, key
    /// and value layers a third each (mflux's `split_q_up` and friends); the down matrix is shared.
    enum Split {
        case none, q, k, v
    }

    struct Row {
        let source: String
        let target: String
        let split: Split

        init(_ source: String, _ target: String, _ split: Split) {
            self.source = source
            self.target = target
            self.split = split
        }
    }

    /// The families' tables, by mflux's mapping.
    public enum Table: String, Sendable {
        case zImage = "Z-Image"
        case flux2 = "FLUX.2"
        case qwen = "Qwen-Image"

        var rows: [Row] {
            switch self {
            case .zImage: LoRAMapping.zImageRows
            case .flux2: LoRAMapping.flux2Rows
            case .qwen: LoRAMapping.qwenRows + LoRAMapping.qwenExtraRows
            }
        }
    }

    /// Qwen-Image's modulation layers, which diffusers names `img_mod.1` and mflux `img_mod_linear`:
    /// mflux's mapping leaves them out (a file that adapts them reaches only the rest there), the
    /// engine applies them as the file's trainer did. Its other layers named as in the checkpoint
    /// (`img_in`, `proj_out`, …) need no row: `LoRAAdapters` takes a path that names a linear layer
    /// of the model as it is.
    static let qwenExtraRows: [Row] = [
        Row("transformer_blocks.{block}.img_mod.1", "transformer_blocks.{block}.img_mod_linear", .none),
        Row("transformer_blocks.{block}.txt_mod.1", "transformer_blocks.{block}.txt_mod_linear", .none),
    ]

    enum Matrix {
        /// `lora_A` (`lora_down`): [rank, in].
        case down
        /// `lora_B` (`lora_up`): [out, rank].
        case up
        case alpha
    }

    let table: Table
    private let rows: [String: [Row]]
    /// A dotted source with underscores for its dots → the source: Kohya's (`lora_unet_…`) and
    /// LyCORIS's (`lycoris_…`) keys are looked up through it.
    private let underscored: [String: String]

    public init(_ table: Table) {
        self.table = table
        rows = Dictionary(grouping: table.rows, by: \.source)
        underscored = Dictionary(rows.keys.map { (Self.underscoredForm($0), $0) }, uniquingKeysWith: { first, _ in first })
    }

    // MARK: Keys

    static let prefixes = ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer."]
    static let underscoredPrefixes = ["lora_unet_", "lycoris_"]
    /// The matrix names mflux's mappings accept, each with or without `.default` and `.weight`.
    private static let matrixSuffixes: [(suffix: String, matrix: Matrix)] = {
        var suffixes: [(String, Matrix)] = []
        for (name, matrix) in [("lora_A", Matrix.down), ("lora_B", .up), ("lora_down", .down), ("lora_up", .up),
                               ("lora.down", .down), ("lora.up", .up)] {
            for tail in [".default.weight", ".weight", ".default", ""] { suffixes.append((".\(name)\(tail)", matrix)) }
        }
        suffixes.append((".alpha", .alpha))
        return suffixes
    }()

    /// The path a key names, without the prefixes and the matrix name, and which matrix it holds;
    /// nil for a key that is not a LoRA matrix (`dora_scale`, LoKr's factors, metadata).
    static func parse(_ key: String) -> (path: String, matrix: Matrix)? {
        guard let (suffix, matrix) = matrixSuffixes.first(where: { key.hasSuffix($0.suffix) }) else { return nil }
        var path = String(key.dropLast(suffix.count))
        var stripped = true
        while stripped {
            stripped = false
            for prefix in prefixes where path.hasPrefix(prefix) {
                path.removeFirst(prefix.count)
                stripped = true
            }
        }
        return (path, matrix)
    }

    static func underscoredForm(_ path: String) -> String {
        path.replacingOccurrences(of: ".", with: "_")
    }

    // MARK: Lookup

    /// The layers a path adapts (`{block}` filled in) and the part of the up matrix each takes;
    /// empty when no row names it.
    func targets(of path: String) -> [(path: String, split: Split)] {
        var underscoredPath: String?
        for prefix in Self.underscoredPrefixes where path.hasPrefix(prefix) {
            underscoredPath = String(path.dropFirst(prefix.count))
        }
        for (template, block) in Self.templates(of: underscoredPath ?? path) {
            let source = underscoredPath == nil ? template : underscored[template]
            guard let source, let matches = rows[source] else { continue }
            return matches.map { row in
                (block.map { row.target.replacingOccurrences(of: "{block}", with: String($0)) } ?? row.target, row.split)
            }
        }
        return []
    }

    /// The path as it is, then with each of its numbers in turn as `{block}`.
    static func templates(of path: String) -> [(template: String, block: Int?)] {
        var templates: [(String, Int?)] = [(path, nil)]
        var index = path.startIndex
        while index < path.endIndex {
            guard path[index].isASCII, path[index].isNumber else {
                index = path.index(after: index)
                continue
            }
            var end = index
            while end < path.endIndex, path[end].isASCII, path[end].isNumber { end = path.index(after: end) }
            if let number = Int(path[index ..< end]) {
                templates.append((path.replacingCharacters(in: index ..< end, with: "{block}"), number))
            }
            index = end
        }
        return templates
    }
}
