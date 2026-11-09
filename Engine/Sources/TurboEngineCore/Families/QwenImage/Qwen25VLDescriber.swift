import Foundation
import MLX
import MLXNN

// Describe Image: Qwen2.5-VL 7B Instruct writes a text-to-image prompt from a picture, as mlx-vlm 0.7.12 runs it
// (`mlx_vlm/models/qwen2_5_vl`, `generate_step`) on mlx-community's 4-bit pack: the picture prepared as its image
// processor prepares it, the vision tower in the weights' float16, its tokens in the chat template's placeholders,
// the language model with its multimodal rotary positions (the picture's tokens at their frame, row and column, the
// text after them from the largest on), greedy decoding with a key–value cache until `<|im_end|>`. The vision tower
// and the language model are Qwen-Image's own modules (Qwen25VisionTower.swift, Qwen25TextEncoder.swift): its text
// encoder is this model's language model, read after the final norm. What is added here is the head, the cache and
// the positions. Names follow the pack (`vision_tower.…`, `language_model.model.…`, `language_model.lm_head`), the
// merger's two linears renamed as mflux names them.

public enum DescribeError: LocalizedError {
    case missingFile(String, URL)
    case unreadablePicture(String)
    case pictureTooThin(Int, Int)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let name, let url): "The describer at \(url.path) has no \(name)."
        case .unreadablePicture(let path): "Couldn’t read the picture \((path as NSString).lastPathComponent)."
        case .pictureTooThin(let width, let height): "A picture of \(width) × \(height) pixels is too long and thin to describe."
        }
    }
}

/// What Describe Image reads of the pack's `config.json`: the language model's and the vision tower's geometry (the
/// 7B's are Qwen-Image's text encoder's and Qwen-Image Edit's tower), the rotary sections and the special tokens.
public struct Qwen25VLConfig: Sendable {
    public var text = QwenImageConfig.TextEncoder()
    public var vision = QwenImageConfig.Vision()
    /// How many of the 64 rotary frequencies turn with the frame, the row and the column of a picture's token.
    public var mropeSection = [16, 24, 24]
    public var visionStartTokenId = 151_652
    /// `<|im_end|>`, which ends the answer; `<|endoftext|>` ends it too.
    public var eosTokenId = 151_645
    public var endOfTextTokenId = 151_643

    static func read(folder: URL) throws -> Qwen25VLConfig {
        let url = folder.appending(path: "config.json")
        guard let data = try? Data(contentsOf: url), let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw DescribeError.missingFile("config.json", folder) }
        func int(_ dict: [String: Any]?, _ key: String, _ fallback: Int) -> Int { (dict?[key] as? NSNumber)?.intValue ?? fallback }
        var config = Qwen25VLConfig()
        config.text.vocabSize = int(json, "vocab_size", config.text.vocabSize)
        config.text.hiddenSize = int(json, "hidden_size", config.text.hiddenSize)
        config.text.numHiddenLayers = int(json, "num_hidden_layers", config.text.numHiddenLayers)
        config.text.numAttentionHeads = int(json, "num_attention_heads", config.text.numAttentionHeads)
        config.text.numKeyValueHeads = int(json, "num_key_value_heads", config.text.numKeyValueHeads)
        config.text.intermediateSize = int(json, "intermediate_size", config.text.intermediateSize)
        config.text.ropeTheta = (json["rope_theta"] as? NSNumber)?.doubleValue ?? config.text.ropeTheta
        config.text.rmsNormEps = (json["rms_norm_eps"] as? NSNumber)?.floatValue ?? config.text.rmsNormEps
        config.text.imageTokenId = int(json, "image_token_id", config.text.imageTokenId)
        config.visionStartTokenId = int(json, "vision_start_token_id", config.visionStartTokenId)
        config.eosTokenId = int(json, "eos_token_id", config.eosTokenId)
        if let section = (json["rope_scaling"] as? [String: Any])?["mrope_section"] as? [NSNumber] {
            config.mropeSection = section.map(\.intValue)
        }
        let vision = json["vision_config"] as? [String: Any]
        config.vision.depth = int(vision, "depth", config.vision.depth)
        config.vision.embedDim = int(vision, "hidden_size", config.vision.embedDim)
        config.vision.mlpHiddenDim = int(vision, "intermediate_size", config.vision.mlpHiddenDim)
        config.vision.numHeads = int(vision, "num_heads", config.vision.numHeads)
        config.vision.patchSize = int(vision, "patch_size", config.vision.patchSize)
        config.vision.temporalPatchSize = int(vision, "temporal_patch_size", config.vision.temporalPatchSize)
        config.vision.spatialMergeSize = int(vision, "spatial_merge_size", config.vision.spatialMergeSize)
        config.vision.windowSize = int(vision, "window_size", config.vision.windowSize)
        if let blocks = vision?["fullatt_block_indexes"] as? [NSNumber] { config.vision.fullAttentionBlocks = blocks.map(\.intValue) }
        return config
    }
}

/// A picture as Describe Image reads it, after mlx-vlm's `load_image` and its Qwen image processor
/// (`Qwen3VLImageProcessor`, which it runs for Qwen2.5-VL too).
public struct Qwen25VLPicture {
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int

    /// The fewest and the most pixels the picture is read at: 256 to 1280 tokens of 28 × 28 pixels, the range Qwen's
    /// model card suggests to balance detail against time and memory (the pack's own allows up to 16,384 tokens).
    public static let minPixels = 256 * 28 * 28
    public static let maxPixels = 1280 * 28 * 28

    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb
        self.width = width
        self.height = height
    }

    /// The image at `path`, turned upright by its EXIF orientation as `ImageOps.exif_transpose` does (the three
    /// rotations: a mirrored orientation, which cameras do not write, stays as stored), on an sRGB canvas. A
    /// transparent picture is laid over white, where mlx-vlm would keep the colours its transparent pixels hide.
    public init(path: String) throws {
        guard let image = ImagePixels.load(path) else { throw DescribeError.unreadablePicture(path) }
        let stored = ImagePixels.crop(ImagePixels.rgbaBytes(image), rgbaWidth: image.width, left: 0, top: 0, width: image.width, height: image.height)
        let (rgb, width, height) = LTXReferenceImage.upright(stored, width: image.width, height: image.height,
                                                             orientation: ImagePixels.orientation(path))
        // `smart_resize` refuses these.
        guard max(width, height) <= 200 * min(width, height) else { throw DescribeError.pictureTooThin(width, height) }
        self.init(rgb: rgb, width: width, height: height)
    }

    /// What the vision tower reads (`_process_one`): resized with Pillow's bicubic to `smart_resize`'s size (sides
    /// multiples of 28, between `minPixels` and `maxPixels`), its passes in Pillow 13's order (what installing mlx-vlm
    /// brings), scaled by 1/255 and normalized with CLIP's mean and deviation in float32 as numpy computes them, in
    /// 14-pixel patches of two identical frames, each 2 × 2 patches in a row. The patches [N, 3·2·14·14] in float32 and
    /// their grid.
    public func visionInput(_ vision: QwenImageConfig.Vision, minPixels: Int = Self.minPixels, maxPixels: Int = Self.maxPixels)
        -> (pixelValues: MLXArray, grid: QwenVisionGrid) {
        let patch = vision.patchSize
        let merge = vision.spatialMergeSize
        let target = QwenEditPicture.smartResize(height: height, width: width, factor: patch * merge, minPixels: minPixels, maxPixels: maxPixels)
        let bytes = PILResample.resizePillow13(rgb, width: width, height: height, toWidth: target.width, toHeight: target.height,
                                               filter: .bicubic)
        // `img * rescale_factor` multiplies by 1/255 rounded to float32, which is not always the same as dividing.
        let scale = Float(1.0 / 255.0)
        var values = [Float](repeating: 0, count: bytes.count)
        for index in 0 ..< bytes.count {
            let channel = index % 3
            values[index] = (Float(bytes[index]) * scale - QwenEditPicture.mean[channel]) / QwenEditPicture.std[channel]
        }
        let image = MLXArray(values, [target.height, target.width, 3]).transposed(2, 0, 1)
        let frames = vision.temporalPatchSize
        let (rows, columns) = (target.height / patch, target.width / patch)
        let patches = stacked(Array(repeating: image, count: frames), axis: 0)
            .reshaped([1, frames, 3, rows / merge, merge, patch, columns / merge, merge, patch])
            .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshaped([rows * columns, 3 * frames * patch * patch])
        return (patches, QwenVisionGrid(t: 1, h: rows, w: columns))
    }
}

/// The language model with its head: `language_model.model` is Qwen-Image's encoder (without its vision tower).
final class Qwen25VLLanguageModel: Module {
    @ModuleInfo(key: "model") var model: Qwen25Encoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(config: QwenImageConfig.TextEncoder) {
        _model.wrappedValue = Qwen25Encoder(config: config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }
}

final class Qwen25VL: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: Qwen25VisionTower
    @ModuleInfo(key: "language_model") var languageModel: Qwen25VLLanguageModel

    init(config: Qwen25VLConfig) {
        _visionTower.wrappedValue = Qwen25VisionTower(config.vision, outDim: config.text.hiddenSize)
        _languageModel.wrappedValue = Qwen25VLLanguageModel(config: config.text)
        super.init()
    }
}

/// What a description came to: the text, the tokens written, and how many tokens the picture took.
public struct Qwen25VLDescription {
    public let text: String
    public let tokens: [Int]
    public let pictureTokens: Int
}

/// Qwen2.5-VL 7B Instruct describing a picture (mlx-community/Qwen2.5-VL-7B-Instruct-4bit: the language model's
/// linears, embeddings and head in 4 bits, the vision tower in float16). Loaded for a description and released after
/// it, beside the image model (`Engine.describe`).
public final class Qwen25VLDescriber {
    public static let family = "qwen2.5-vl"
    /// The most tokens a description may take; the instruction asks for one paragraph, which takes 100 to 200.
    public static let maxTokens = 300

    /// What the picture's description is asked for, after it in the user's turn. Tried on photos, paintings, a
    /// poster and close-ups: naming kinds of picture as examples makes the model take them for this one (a photo
    /// called a watercolor), asking for the text it reads in quotes makes it invent some where there is none, and
    /// without the last sentence most answers start with "The image is".
    public static let instruction = "Write a prompt for a text-to-image model that would re-create this picture: one paragraph that "
        + "starts with the kind of picture and its main subject, then describes the setting, the composition, the lighting and "
        + "the colors. If the picture contains text, copy it exactly as written, in double quotes. Write only the prompt, and "
        + "do not start it with \"The picture\" or \"The image\"."

    public let config: Qwen25VLConfig
    let model: Qwen25VL
    let tokenizer: BPETokenizer

    /// Loads the pack at `folder`, its weights read into memory at once (5.7 GB).
    public init(folder: URL) throws {
        for name in ["config.json", "model.safetensors.index.json", "tokenizer.json"]
        where !FileManager.default.fileExists(atPath: folder.appending(path: name).path) {
            throw DescribeError.missingFile(name, folder)
        }
        config = try Qwen25VLConfig.read(folder: folder)
        model = Qwen25VL(config: config)
        var weights: [String: MLXArray] = [:]
        for shard in try Checkpoint.shards(in: folder, component: "") {
            for (key, value) in try loadArrays(url: shard) { weights[Self.moduleKey(key)] = value }
        }
        try WeightLoading.apply(weights, to: model)
        eval(model)
        tokenizer = try BPETokenizer(folder: folder)
    }

    /// The pack's merger keeps mlx-vlm's list of layers (`mlp.0`, a GELU, `mlp.2`), named as mflux names them here.
    static func moduleKey(_ key: String) -> String {
        key.replacingOccurrences(of: "vision_tower.merger.mlp.0.", with: "vision_tower.merger.mlp_0.")
            .replacingOccurrences(of: "vision_tower.merger.mlp.2.", with: "vision_tower.merger.mlp_1.")
    }

    // MARK: The prompt

    /// The chat template's text for a picture of `pictureTokens` tokens and the instruction, as mlx-vlm's
    /// `apply_chat_template` writes it for Qwen2.5-VL (the picture first, the default system prompt), with the
    /// picture's placeholder repeated as its processor repeats it.
    public static func text(pictureTokens: Int, instruction: String = instruction) -> String {
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>"
            + String(repeating: "<|image_pad|>", count: pictureTokens)
            + "<|vision_end|>\(instruction)<|im_end|>\n<|im_start|>assistant\n"
    }

    public func tokenIds(pictureTokens: Int, instruction: String = instruction) -> [Int] {
        tokenIds(text: Self.text(pictureTokens: pictureTokens, instruction: instruction))
    }

    /// Any text's ids, the special tokens it names among them, as the processor tokenizes the prompt.
    public func tokenIds(text: String) -> [Int] {
        tokenizer.encode(text, addSpecialTokens: false)
    }

    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens)
    }

    /// `get_rope_index` for one picture: the text before it at positions 0, 1, 2… on all three axes; each of the
    /// picture's tokens at the position of its first one plus its frame, row and column (of merged tokens); the text
    /// after it from the largest of those plus one. Returns the positions [3][S] and how far the next one is from the
    /// sequence's length (`rope_deltas`: below it, a picture taking fewer positions than tokens), which the written
    /// tokens' positions keep.
    public func positions(ids: [Int], grid: QwenVisionGrid) -> (positions: [[Int]], delta: Int) {
        let merge = config.vision.spatialMergeSize
        let (rows, columns) = (grid.h / merge, grid.w / merge)
        var positions: [[Int]] = [[], [], []]
        guard let start = ids.firstIndex(of: config.text.imageTokenId) else {
            positions = Array(repeating: Array(0 ..< ids.count), count: 3)
            return (positions, 0)
        }
        for axis in 0 ..< 3 { positions[axis] = Array(0 ..< start) }
        for frame in 0 ..< grid.t {
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    positions[0].append(start + frame)
                    positions[1].append(start + row)
                    positions[2].append(start + column)
                }
            }
        }
        let next = start + max(grid.t, rows, columns)
        let rest = ids.count - positions[0].count
        for axis in 0 ..< 3 { positions[axis] += Array(next ..< next + rest) }
        return (positions, next + rest - ids.count)
    }

    // MARK: The model

    /// The vision tower's tokens for the picture: [N / 4, hidden] in the weights' float16, as mlx-vlm casts the pixels.
    public func pictureEmbeds(pixelValues: MLXArray, grid: QwenVisionGrid) -> MLXArray {
        model.visionTower(pixelValues, grids: [grid], dtype: model.visionTower.patchEmbed.proj.weight.dtype)
    }

    /// The rotary angles of `positions` [3][L], each frequency turned by the axis its section belongs to (`chunked`
    /// sections: the first 16 by the frame, the next 24 by the row, the last 24 by the column), cos and sin
    /// [1, L, headDim] in float32, as mlx-vlm's fused kernel computes and uses them.
    func rotary(_ positions: [[Int]]) -> (MLXArray, MLXArray) {
        let dim = config.text.headDim
        var axisOf: [Int] = []
        for (axis, count) in config.mropeSection.enumerated() { axisOf += Array(repeating: axis, count: count) }
        let length = positions[0].count
        var angles = [Float](repeating: 0, count: length * axisOf.count)
        for t in 0 ..< length {
            for (frequency, axis) in axisOf.enumerated() { angles[t * axisOf.count + frequency] = Float(positions[axis][t]) }
        }
        let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim)
        let inverse = 1 / pow(Float(config.text.ropeTheta), exponents)
        let freqs = MLXArray(angles, [length, axisOf.count]) * inverse
        let emb = concatenated([freqs, freqs], axis: -1).reshaped([1, length, dim])
        return (cos(emb), sin(emb))
    }

    /// The language model over `embeddings` [1, L, hidden] at `positions`, each layer's keys and values added to its
    /// cache: the final norm's hidden states [1, L, hidden]. Several positions only from an empty cache (the prompt).
    func hidden(_ embeddings: MLXArray, positions: [[Int]], caches: [Music3KVCache]) -> MLXArray {
        let length = embeddings.dim(1)
        let (cosines, sines) = rotary(positions)
        var mask: MLXArray?
        if length > 1 {
            let index = MLXArray(0 ..< Int32(length))
            let later = index.expandedDimensions(axis: 0) .> index.expandedDimensions(axis: 1)
            mask = MLX.where(later, MLXArray(-Float.infinity), MLXArray(Float(0))).reshaped([1, 1, length, length])
        }
        let language = model.languageModel.model
        var h = embeddings
        for (layer, cache) in zip(language.layers, caches) {
            h = layer(h, mask: mask, cos: cosines, sin: sines, cache: cache)
        }
        return language.norm(h)
    }

    /// The greedy choice from the last position's logits, taken from the log-probabilities as mlx-vlm's sampler
    /// takes it (a tie, rare, goes the same way).
    static func greedy(_ logits: MLXArray) -> MLXArray {
        argMax(logits - logSumExp(logits, axis: -1, keepDims: true), axis: -1)
    }

    /// Writes the answer to the prompt `ids` with the picture's tokens `pictureEmbeds` in its placeholders, as
    /// `generate_step` does at temperature 0: the prompt but its last token into the cache, then that token, then
    /// one written token after the other, the next one's step built while the previous one is read, until
    /// `<|im_end|>` or `maxTokens`. `firstLogits` gets the first step's logits [vocab] in float32; `written` the
    /// tokens so far, after each one.
    public func generate(
        ids: [Int], pictureEmbeds: MLXArray, grid: QwenVisionGrid, maxTokens: Int = maxTokens,
        firstLogits: ((MLXArray) -> Void)? = nil, written: ([Int]) -> Void = { _ in }, isCancelled: () -> Bool = { false }
    ) throws -> [Int] {
        let language = model.languageModel
        let inputIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let embeddings = Qwen25Encoder.placing(pictureEmbeds, in: language.model.embedTokens(inputIds), inputIds: inputIds,
                                               tokenId: config.text.imageTokenId)
        let (positions, delta) = self.positions(ids: ids, grid: grid)
        let caches = (0 ..< config.text.numHiddenLayers).map { _ in Music3KVCache() }
        let count = ids.count
        if count > 1 {
            let prompt = hidden(embeddings[0..., 0 ..< count - 1], positions: positions.map { Array($0[0 ..< count - 1]) }, caches: caches)
            eval(prompt)
            if isCancelled() { throw GenerationError.cancelled }
        }
        let last = hidden(embeddings[0..., (count - 1)...], positions: positions.map { [$0[count - 1]] }, caches: caches)
        let logits = language.lmHead(last)
        if let firstLogits {
            let first = logits.reshaped([-1]).asType(.float32)
            eval(first)
            firstLogits(first)
        }
        var token = Self.greedy(logits.reshaped([1, -1]))
        asyncEval(token)
        var tokens: [Int] = []
        let stops: Set<Int> = [config.eosTokenId, config.endOfTextTokenId]
        while tokens.count < maxTokens {
            // The written token's position: the sequence's length so far, plus the picture's lead.
            let position = count + tokens.count + delta
            let embedding = language.model.embedTokens(token.reshaped([1, 1]))
            let step = language.lmHead(hidden(embedding, positions: [[position], [position], [position]], caches: caches))
            let next = Self.greedy(step.reshaped([1, -1]))
            asyncEval(next)
            let id = token.item(Int.self)
            if stops.contains(id) { break }
            tokens.append(id)
            written(tokens)
            if isCancelled() { throw GenerationError.cancelled }
            token = next
        }
        return tokens
    }

    /// The picture described: its tokens read, the prompt with them, the answer written (`text` gets the text so far
    /// whenever it grows by whole characters), the text trimmed. `writing` is called once the first token is written.
    public func describe(
        _ picture: Qwen25VLPicture, instruction: String = instruction, maxTokens: Int = maxTokens,
        writing: () -> Void = {}, text: (String) -> Void = { _ in }, isCancelled: () -> Bool = { false }
    ) throws -> Qwen25VLDescription {
        let (pixelValues, grid) = picture.visionInput(config.vision)
        let embeds = pictureEmbeds(pixelValues: pixelValues, grid: grid)
        eval(embeds)
        if isCancelled() { throw GenerationError.cancelled }
        let pictureTokens = embeds.dim(0)
        var shown = ""
        let tokens = try generate(
            ids: tokenIds(pictureTokens: pictureTokens, instruction: instruction), pictureEmbeds: embeds, grid: grid,
            maxTokens: maxTokens,
            written: { tokens in
                if tokens.count == 1 { writing() }
                // A character cut between two tokens waits for the second.
                let decoded = decode(tokens)
                guard !decoded.hasSuffix("\u{FFFD}"), decoded != shown else { return }
                shown = decoded
                text(decoded.trimmingCharacters(in: .whitespacesAndNewlines))
            },
            isCancelled: isCancelled
        )
        return Qwen25VLDescription(text: decode(tokens).trimmingCharacters(in: .whitespacesAndNewlines), tokens: tokens,
                                   pictureTokens: pictureTokens)
    }
}
