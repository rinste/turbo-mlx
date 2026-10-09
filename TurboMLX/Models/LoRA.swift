import Foundation

/// A LoRA the controls hold for a family: a file of the LoRA library (`LoRALibrary`) and how
/// strongly it applies.
nonisolated struct LoRAChoice: Codable, Hashable, Sendable {
    static let scaleRange = 0.0...2.0

    /// The file's name in the library folder.
    var file: String
    /// 1: as trained; less tones it down, more pushes it further.
    var scale = 1.0

    var name: String { (file as NSString).deletingPathExtension }

    /// "×1", "×0.75".
    var scaleLabel: String {
        "×" + scale.formatted(.number.precision(.fractionLength(0...2)))
    }
}

/// The model a LoRA was trained on, as far as its file tells: the names and sizes of the layers it
/// adapts, or the trainer's note in its metadata.
nonisolated enum LoRABase: Hashable, Sendable {
    case zImage
    /// FLUX.2 Klein, with the transformer's width when the file shows it: 3072 for 4B, 4096 for 9B.
    case flux2(width: Int?)
    case qwenImage
    /// LTX-2.3 and LTX-2.5 (4096 wide); another width is LTX-Video 0.9 or another video model
    /// keyed the same way.
    case ltx(width: Int?)
    case flux1
    case stableDiffusion
    case unknown

    var displayName: String {
        switch self {
        case .zImage: "Z-Image"
        case .flux2(let width): width == 4096 ? "FLUX.2 Klein 9B" : width == 3072 ? "FLUX.2 Klein 4B" : "FLUX.2 Klein"
        case .qwenImage: "Qwen-Image"
        case .ltx(let width): width == nil || width == 4096 ? "LTX-2" : "an older LTX-Video"
        case .flux1: "FLUX.1"
        case .stableDiffusion: "Stable Diffusion"
        case .unknown: "an unknown model"
        }
    }

    /// Whether a model of the app can run it: nil when the file does not say what it is for.
    func fits(_ model: ModelDescriptor) -> Bool? {
        switch self {
        case .zImage: model.family == .zImageTurbo
        case .flux2(let width): model.family == .flux2Klein && (width == nil || width == model.kleinWidth)
        // Qwen-Image's editor has its transformer: a LoRA of one is a LoRA of the other.
        case .qwenImage: model.family == .qwenImage || model.family == .qwenImageEdit
        // LTX-2.3's and LTX-2.5's transformers have the same layers: which one a LoRA was made
        // for shows in its results, not its shapes.
        case .ltx(let width): model.family.isLTX && (width == nil || width == 4096)
        case .flux1, .stableDiffusion: false
        case .unknown: nil
        }
    }

    /// Whether any model the app knows of could run it.
    var isSupported: Bool {
        switch self {
        case .zImage, .flux2, .qwenImage, .unknown: true
        case .ltx(let width): width == nil || width == 4096
        case .flux1, .stableDiffusion: false
        }
    }
}

/// What a LoRA file says about itself, read from its safetensors header: the model it is for, the
/// words its training captions used for what it teaches, how many layers it adapts.
nonisolated struct LoRAFileInfo: Hashable, Sendable {
    var base: LoRABase
    var triggerWords: [String]
    var layers: Int

    enum ReadError: LocalizedError {
        case unreadable
        case notLoRA

        var errorDescription: String? {
            switch self {
            case .unreadable: "The file is not a safetensors file."
            case .notLoRA: "The file holds no LoRA matrices: it may be a whole model rather than a LoRA."
            }
        }
    }

    static func read(_ url: URL) throws -> LoRAFileInfo {
        let (tensors, metadata) = try header(of: url)
        var downs: [String: [Int]] = [:]
        var ups: [String: [Int]] = [:]
        for (key, shape) in tensors {
            guard let (path, matrix) = parse(key) else { continue }
            if matrix == .down { downs[path] = shape } else if matrix == .up { ups[path] = shape }
        }
        guard !downs.isEmpty, !ups.isEmpty else { throw ReadError.notLoRA }
        return LoRAFileInfo(base: base(downs: downs, ups: ups, metadata: metadata), triggerWords: triggerWords(metadata),
                            layers: downs.count)
    }

    // MARK: Header

    /// The tensors' names and shapes and the metadata: the JSON after the file's first 8 bytes.
    static func header(of url: URL) throws -> (tensors: [String: [Int]], metadata: [String: String]) {
        guard let handle = try? FileHandle(forReadingFrom: url) else { throw ReadError.unreadable }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8), prefix.count == 8 else { throw ReadError.unreadable }
        let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard length > 0, length < 64 << 20, let data = try? handle.read(upToCount: Int(length)), data.count == Int(length),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw ReadError.unreadable }
        var tensors: [String: [Int]] = [:]
        var metadata: [String: String] = [:]
        for (key, value) in json {
            if key == "__metadata__" {
                metadata = (value as? [String: Any])?.compactMapValues { $0 as? String } ?? [:]
            } else if let entry = value as? [String: Any], let shape = entry["shape"] as? [Int] {
                tensors[key] = shape
            }
        }
        return (tensors, metadata)
    }

    // MARK: Keys

    enum Matrix {
        case down, up, alpha
    }

    /// The layer path a key names and its matrix, the way the engine reads it
    /// (`LoRAMapping.parse`): the trainer's prefixes and the matrix names stripped.
    static func parse(_ key: String) -> (path: String, matrix: Matrix)? {
        let names: [(String, Matrix)] = [("lora_A", .down), ("lora_B", .up), ("lora_down", .down), ("lora_up", .up),
                                         ("lora.down", .down), ("lora.up", .up)]
        var found: (path: String, matrix: Matrix)?
        for (name, matrix) in names {
            for tail in [".default.weight", ".weight", ".default", ""] where key.hasSuffix(".\(name)\(tail)") {
                found = (String(key.dropLast(name.count + tail.count + 1)), matrix)
                break
            }
            if found != nil { break }
        }
        if found == nil, key.hasSuffix(".alpha") { found = (String(key.dropLast(6)), .alpha) }
        guard var (path, matrix) = found else { return nil }
        var stripped = true
        while stripped {
            stripped = false
            for prefix in ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer."] where path.hasPrefix(prefix) {
                path.removeFirst(prefix.count)
                stripped = true
            }
        }
        // Kohya's and LyCORIS's spellings, with underscores for the dots, read as dotted paths for
        // the checks below (good enough to tell the families apart).
        for prefix in ["lora_unet_", "lycoris_"] where path.hasPrefix(prefix) {
            path = String(path.dropFirst(prefix.count)).replacingOccurrences(of: "_", with: ".")
        }
        return (path, matrix)
    }

    // MARK: Base model

    static func base(downs: [String: [Int]], ups: [String: [Int]], metadata: [String: String]) -> LoRABase {
        let width = attentionWidth(downs)
        let note = [metadata["ss_base_model_version"], metadata["modelspec.architecture"], metadata["base_model"]]
            .compactMap { $0?.lowercased() }.joined(separator: " ")
        if note.contains("zimage") || note.contains("z-image") || note.contains("z_image") { return .zImage }
        if note.contains("qwen") { return .qwenImage }
        if note.contains("ltx") { return .ltx(width: ltxWidth(downs)) }
        if note.contains("flux2") || note.contains("flux.2") || note.contains("flux-2") || note.contains("flux_2") { return .flux2(width: width) }
        if note.contains("flux1") || note.contains("flux.1") || note.contains("flux-1") || note.contains("flux_1") { return .flux1 }
        if note.contains("sdxl") || note.contains("sd_xl") || note.contains("stable-diffusion") || note.contains("sd_v1") { return .stableDiffusion }

        // Underscored keys were turned into dots above, so names are matched loosely.
        let paths = Array(downs.keys)
        func any(_ test: (String) -> Bool) -> Bool { paths.contains(where: test) }
        if any({ $0.hasPrefix("layers.") || $0.hasPrefix("noise.refiner") || $0.hasPrefix("context.refiner") || $0.hasPrefix("noise_refiner") || $0.hasPrefix("context_refiner") }) {
            return .zImage
        }
        if any({ $0.hasPrefix("down.blocks") || $0.hasPrefix("up.blocks") || $0.hasPrefix("mid.block") || $0.hasPrefix("down_blocks") || $0.hasPrefix("up_blocks") || $0.hasPrefix("mid_block") || $0.hasPrefix("te.") || $0.hasPrefix("te1.") || $0.hasPrefix("te2.") || $0.hasPrefix("text_encoder") }) {
            return .stableDiffusion
        }
        // LTX's audio–video blocks, before FLUX.1, whose feed-forward is named alike (`ff.net`).
        let ltxMarks = ["audio_attn", "audio.attn", "video_to_audio", "video.to.audio", "audio_to_video", "audio.to.video",
                        "audio_ff", "audio.ff", "patchify_proj", "patchify.proj", "adaln_single", "adaln.single"]
        if any({ path in ltxMarks.contains { path.contains($0) } })
            || any({ ($0.hasPrefix("transformer_blocks.") || $0.hasPrefix("transformer.blocks.")) && ($0.contains(".attn1.") || $0.contains(".attn2.")) }) {
            return .ltx(width: ltxWidth(downs))
        }
        if any({ $0.hasPrefix("transformer") && ($0.contains("img.mlp") || $0.contains("img_mlp") || $0.contains("txt_mlp") || $0.contains("txt.mlp") || $0.contains("img_mod") || $0.contains("img.mod") || $0.contains("txt_mod")) }) {
            return .qwenImage
        }
        let flux2Marks = ["qkv.mlp.proj", "qkv_mlp_proj", "stream.modulation", "stream_modulation", "time.guidance", "time_guidance", "linear.in", "linear_in"]
        if any({ path in flux2Marks.contains { path.contains($0) } }) { return .flux2(width: width) }
        let flux1Marks = ["proj.mlp", "proj_mlp", "norm1.linear", "norm1_context", "norm1.context", "ff.net", "ff_context.net", "img.mod", "txt.mod", "img_mod.lin", "modulation.lin", "vector.in", "vector_in", "guidance.in", "guidance_in"]
        if any({ path in flux1Marks.contains { path.contains($0) } }) { return .flux1 }
        if any({ $0.contains("single") && ($0.hasSuffix("to.q") || $0.hasSuffix("to_q") || $0.hasSuffix("proj.out") || $0.hasSuffix("proj_out")) }) {
            return .flux1
        }
        // BFL's single block: one projection for the query, key, value and MLP, whose size tells
        // FLUX.1 (3 × 3072 + 4 × 3072) from FLUX.2 (3 × width + 6 × width).
        if let rows = ups.first(where: { $0.key.contains("single") && $0.key.hasSuffix("linear1") })?.value.first {
            if rows == 21504 { return .flux1 }
            if rows == 27648 || rows == 36864 { return .flux2(width: rows / 9) }
        }
        // Only attention in double-stream blocks: Qwen-Image has 60, FLUX.1 19, FLUX.2 Klein 5 or 8.
        let blocks = paths.compactMap { path -> Int? in
            let parts = path.split(separator: ".")
            guard let index = parts.firstIndex(where: { $0 == "blocks" || $0.hasSuffix("blocks") }), index + 1 < parts.count else { return nil }
            return Int(parts[index + 1])
        }
        if let last = blocks.max(), last >= 19 { return .qwenImage }
        if width == 4096 { return .flux2(width: 4096) }
        return .unknown
    }

    /// LTX's video width, from the input size of the video self-attention's query.
    static func ltxWidth(_ downs: [String: [Int]]) -> Int? {
        downs.first { ($0.key.hasSuffix(".attn1.to_q") || $0.key.hasSuffix(".attn1.to.q")) && !$0.key.contains("audio") }?.value.last
    }

    /// The transformer's width, from the input size of an attention projection.
    static func attentionWidth(_ downs: [String: [Int]]) -> Int? {
        let ends = ["to_q", "to.q", "to_k", "to.k", "qkv", "to_qkv_mlp_proj", "qkv.mlp.proj"]
        return downs.first { entry in ends.contains { entry.key.hasSuffix($0) } }?.value.last
    }

    // MARK: Trigger words

    /// The words to put in a prompt: the trainer's trigger phrase, or the captions' tags (Kohya's
    /// and ai-toolkit's `ss_tag_frequency`), the most frequent first.
    static func triggerWords(_ metadata: [String: String]) -> [String] {
        for key in ["modelspec.trigger_phrase", "trigger_phrase", "trigger_word", "trigger_words", "ss_trigger_words", "instance_prompt"] {
            if let value = metadata[key]?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
                return value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            }
        }
        guard let data = metadata["ss_tag_frequency"]?.data(using: .utf8),
              let folders = try? JSONSerialization.jsonObject(with: data) as? [String: [String: Int]]
        else { return [] }
        var counts: [String: Int] = [:]
        for tags in folders.values {
            for (tag, count) in tags {
                let word = tag.trimmingCharacters(in: .whitespaces)
                if !word.isEmpty { counts[word, default: 0] += count }
            }
        }
        return counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.prefix(3).map(\.key)
    }
}
