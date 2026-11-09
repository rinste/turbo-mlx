import Foundation
import ImageIO

/// The prompt and settings a picture says it was made with, read back from its file: Turbo MLX's own
/// PNGs, and those of the apps people often come from, AUTOMATIC1111 (and Forge and SD.Next, which
/// write as it does), InvokeAI and ComfyUI. Another app's model is not one of ours: only a Turbo MLX
/// picture names a model to choose.
nonisolated struct ImageSettings: Equatable, Sendable {
    /// A LoRA a Turbo MLX picture was made with, by the name of its file without the extension.
    struct LoRA: Equatable, Sendable {
        var name: String
        var scale: Double
    }

    var prompt: String
    var negativePrompt: String?
    var seed: Int?
    var steps: Int?
    var guidance: Double?
    var size: PixelSize?
    /// Turbo MLX only: the model (`ModelDescriptor.id`), its LoRAs, the 16-bit precision and, for a
    /// clip's first frame, LTX's Quality mode.
    var modelID: String?
    var loras: [LoRA] = []
    var halfPrecision = false
    var quality = false

    /// What the picture at `url` says, or nil when it says nothing Turbo MLX can use.
    static func read(_ url: URL) -> ImageSettings? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        else { return nil }
        // The engine's JSON, in the description ImageIO keeps in the PNG's XMP.
        let png = properties[kCGImagePropertyPNGDictionary] as? [CFString: Any]
        if let description = png?[kCGImagePropertyPNGDescription] as? String, let found = turboMLX(description) {
            return found
        }
        let chunks = pngText(url)
        if let text = chunks["parameters"], let found = automatic1111(text) { return found }
        if let text = chunks["invokeai_metadata"], let found = invokeAI(text) { return found }
        if let text = chunks["prompt"], let found = comfyUI(text) { return found }
        // AUTOMATIC1111 keeps the same text in the Exif user comment of a JPEG or a WebP; there a
        // comment must have its settings line to be taken for one.
        let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any]
        if let comment = exif?[kCGImagePropertyExifUserComment] as? String, let found = automatic1111(comment), found.steps != nil {
            return found
        }
        return nil
    }

    // MARK: Turbo MLX

    /// The JSON the engine writes into the description (`ImageOutput.writePNG`), which names it.
    static func turboMLX(_ description: String) -> ImageSettings? {
        guard let json = object(description), (json["engine"] as? String)?.hasPrefix("turbo-engine") == true,
              let prompt = json["prompt"] as? String
        else { return nil }
        var found = ImageSettings(prompt: prompt)
        found.negativePrompt = json["negative_prompt"] as? String
        found.seed = seed(json["seed"])
        found.steps = json["steps"] as? Int
        found.guidance = json["guidance"] as? Double
        found.size = size(json["width"], json["height"])
        found.modelID = json["model"] as? String
        found.loras = (json["loras"] as? [[String: Any]] ?? []).compactMap { lora in
            guard let name = lora["name"] as? String else { return nil }
            return LoRA(name: name, scale: lora["scale"] as? Double ?? 1)
        }
        found.halfPrecision = json["precision"] as? String == "bf16"
        found.quality = json["quality"] as? Bool ?? false
        return found
    }

    // MARK: AUTOMATIC1111

    /// AUTOMATIC1111's text, read as its `parse_generation_parameters` (modules/infotext_utils.py)
    /// reads it: the prompt, over as many lines as it takes; from a line that starts with "Negative
    /// prompt:", the negative prompt; and a last line of "Key: value" pairs, if it holds at least
    /// three, where a value with a comma in it is in JSON quotes. The prompt keeps its attention
    /// syntax, "(word:1.2)", as written. Fooocus can write JSON under the same keyword: that is not
    /// a prompt.
    static func automatic1111(_ text: String) -> ImageSettings? {
        guard object(text) == nil else { return nil }
        var lines = text.replacingOccurrences(of: "\r\n", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var fields: [String: String] = [:]
        if let last = lines.last {
            let pairs = last.matches(of: /\s*(\w[\w \-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)/)
            if pairs.count >= 3 {
                lines.removeLast()
                for pair in pairs {
                    let value = String(pair.output.2)
                    fields[String(pair.output.1)] = value.hasPrefix("\"") && value.hasSuffix("\"")
                        ? (try? JSONSerialization.jsonObject(with: Data(value.utf8), options: .fragmentsAllowed)) as? String ?? value
                        : value
                }
            }
        }
        var prompt = ""
        var negative: String?
        for line in lines {
            var line = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.hasPrefix("Negative prompt:") {
                negative = negative ?? ""
                line = line.dropFirst("Negative prompt:".count).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if let text = negative {
                negative = text.isEmpty ? line : text + "\n" + line
            } else {
                prompt = prompt.isEmpty ? line : prompt + "\n" + line
            }
        }
        let field = { (key: String) in fields[key]?.trimmingCharacters(in: .whitespaces) }
        var found = ImageSettings(prompt: prompt)
        found.negativePrompt = negative
        found.seed = seed(field("Seed").flatMap { Int($0) })
        found.steps = field("Steps").flatMap { Int($0) }
        found.guidance = field("CFG scale").flatMap { Double($0) }
        if let sides = field("Size")?.wholeMatch(of: /(\d+)x(\d+)/) {
            found.size = size(Int(sides.output.1), Int(sides.output.2))
        }
        guard !found.prompt.isEmpty || !fields.isEmpty else { return nil }
        return found
    }

    // MARK: InvokeAI

    /// InvokeAI's `invokeai_metadata`: the parameters of the generation as JSON.
    static func invokeAI(_ text: String) -> ImageSettings? {
        guard let json = object(text), let prompt = json["positive_prompt"] as? String else { return nil }
        var found = ImageSettings(prompt: prompt)
        found.negativePrompt = json["negative_prompt"] as? String
        found.seed = seed(json["seed"])
        found.steps = json["steps"] as? Int
        found.guidance = json["cfg_scale"] as? Double
        found.size = size(json["width"], json["height"])
        return found
    }

    // MARK: ComfyUI

    /// ComfyUI's `prompt`: the graph that ran, node by node. Read only when it is plain: a single
    /// sampler whose prompts come straight from text encoders with their text written in, and whose
    /// latent is an empty one of a given size. Anything else, a refiner pass or a prompt built by other
    /// nodes, could say a prompt that is not the picture's: nothing is taken from it.
    static func comfyUI(_ text: String) -> ImageSettings? {
        guard let graph = object(text)?.compactMapValues({ $0 as? [String: Any] }) else { return nil }
        let samplers = graph.values.filter { ["KSampler", "KSamplerAdvanced"].contains($0["class_type"] as? String) }
        guard samplers.count == 1, let inputs = samplers[0]["inputs"] as? [String: Any] else { return nil }
        // The node an input is wired to, given as [node id, output].
        let source = { (input: String) -> [String: Any]? in
            ((inputs[input] as? [Any])?.first as? String).flatMap { graph[$0] }
        }
        let encoded = { (input: String) -> String? in
            guard let node = source(input), node["class_type"] as? String == "CLIPTextEncode" else { return nil }
            return (node["inputs"] as? [String: Any])?["text"] as? String
        }
        guard let prompt = encoded("positive") else { return nil }
        var found = ImageSettings(prompt: prompt)
        found.negativePrompt = encoded("negative")
        found.seed = seed(inputs["seed"] ?? inputs["noise_seed"])
        found.steps = inputs["steps"] as? Int
        found.guidance = inputs["cfg"] as? Double
        if let latent = source("latent_image"), ["EmptyLatentImage", "EmptySD3LatentImage"].contains(latent["class_type"] as? String),
           let size = latent["inputs"] as? [String: Any] {
            found.size = Self.size(size["width"], size["height"])
        }
        return found
    }

    // MARK: Values

    private static func object(_ text: String) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any]
    }

    /// A seed the controls can keep: not negative (the engine makes an unsigned number of it), and
    /// with room to count up a batch from it.
    private static func seed(_ value: Any?) -> Int? {
        guard let seed = value as? Int, (0...(Int.max - GenerationSettings.maxBatch)).contains(seed) else { return nil }
        return seed
    }

    private static func size(_ width: Any?, _ height: Any?) -> PixelSize? {
        guard let width = width as? Int, let height = height as? Int, width > 0, height > 0 else { return nil }
        return PixelSize(width: width, height: height)
    }

    // MARK: PNG text

    /// A PNG's text chunks by keyword (the first of each): ImageIO hands over only the few keywords
    /// it knows (Description, Comment…), not those other apps keep their settings under. tEXt is
    /// Latin-1; zTXt is Latin-1 compressed; iTXt is UTF-8, compressed or not. The image data is
    /// skipped, not read. Nothing for a file that is not a PNG.
    static func pngText(_ url: URL) -> [String: String] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return [:] }
        defer { try? handle.close() }
        guard (try? handle.read(upToCount: 8)) == Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) else { return [:] }
        var chunks: [String: String] = [:]
        var offset: UInt64 = 8
        while let header = try? handle.read(upToCount: 8), header.count == 8 {
            let length = header.prefix(4).reduce(0) { $0 << 8 | Int($1) }
            let type = String(decoding: header.suffix(4), as: UTF8.self)
            if type == "IEND" { break }
            // Text chunks hold a few kilobytes, ComfyUI's graphs a few hundred.
            if ["tEXt", "zTXt", "iTXt"].contains(type), length < 16 << 20 {
                guard let data = try? handle.read(upToCount: length), data.count == length else { break }
                if let (keyword, text) = text(of: data, type: type), chunks[keyword] == nil { chunks[keyword] = text }
            }
            // Past the data and its CRC.
            offset += 8 + UInt64(length) + 4
            guard (try? handle.seek(toOffset: offset)) != nil else { break }
        }
        return chunks
    }

    private static func text(of data: Data, type: String) -> (keyword: String, text: String)? {
        let bytes = [UInt8](data)
        guard let end = bytes.firstIndex(of: 0), let keyword = String(bytes: bytes[..<end], encoding: .isoLatin1) else { return nil }
        let rest = bytes[(end + 1)...]
        switch type {
        case "tEXt":
            return String(bytes: rest, encoding: .isoLatin1).map { (keyword, $0) }
        case "zTXt":
            // A compression method (0, zlib), then the stream.
            return inflate(rest.dropFirst()).flatMap { String(data: $0, encoding: .isoLatin1) }.map { (keyword, $0) }
        default:
            // Whether it is compressed, how, then a language tag and a translated keyword, each ending in a zero.
            guard rest.count >= 2 else { return nil }
            let compressed = rest[rest.startIndex] == 1
            let tags = rest.dropFirst(2)
            guard let language = tags.firstIndex(of: 0), let translated = tags[(language + 1)...].firstIndex(of: 0) else { return nil }
            let payload = tags[(translated + 1)...]
            let utf8 = compressed ? inflate(payload) : Data(payload)
            return utf8.flatMap { String(data: $0, encoding: .utf8) }.map { (keyword, $0) }
        }
    }

    /// A zlib stream's data. Foundation's zlib is raw deflate: the stream's two-byte header goes first.
    private static func inflate(_ stream: ArraySlice<UInt8>) -> Data? {
        guard stream.count > 2 else { return nil }
        return try? (Data(stream.dropFirst(2)) as NSData).decompressed(using: .zlib) as Data
    }
}
