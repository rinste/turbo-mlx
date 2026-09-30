import Foundation
import MLX
import MLXNN

/// The tokenizer (the pack's `tokenizer.json`) and the queries the model reads: a prompt's, an
/// edit's with its picture's tokens, and the unconditional ones of guidance, all tokenized as
/// `tokenizer(query)` does (no tokens added).
public final class SenseNovaPrompter {
    private let tokenizer: BPETokenizer
    /// `<img>`, `</img>` and `<IMG_CONTEXT>`, the tokens around and in place of a picture.
    public let imageStart: Int
    public let imageEnd: Int
    public let imageContext: Int

    public init(folder: URL) throws {
        guard FileManager.default.fileExists(atPath: folder.appending(path: "tokenizer.json").path) else {
            throw Qwen3Prompter.PromptError.noTokenizer(folder)
        }
        tokenizer = try BPETokenizer(folder: folder)
        guard let start = tokenizer.id(of: "<img>"), let end = tokenizer.id(of: "</img>"), let context = tokenizer.id(of: "<IMG_CONTEXT>") else {
            throw BPETokenizer.TokenizerError.unsupported("a vocabulary without SenseNova's picture tokens")
        }
        (imageStart, imageEnd, imageContext) = (start, end, context)
    }

    public func tokenIds(_ prompt: String) -> [Int] {
        tokenizer.encode(SenseNovaConfig.conditionalQuery(prompt), addSpecialTokens: true)
    }

    public var unconditionalIds: [Int] {
        tokenizer.encode(SenseNovaConfig.unconditionalQuery, addSpecialTokens: true)
    }

    /// An edit's query with one picture of `pictureTokens` tokens.
    public func editIds(_ prompt: String, pictureTokens: Int) -> [Int] {
        withPicture(SenseNovaConfig.conditionalQuery(SenseNovaConfig.editPrompt(prompt)), tokens: pictureTokens)
    }

    /// The unconditional query of an edit's guidance, with the picture.
    public func editUnconditionalIds(pictureTokens: Int) -> [Int] {
        withPicture(SenseNovaConfig.editUnconditionalQuery, tokens: pictureTokens)
    }

    /// The query with its first `<image>` replaced by `<img>`, `tokens` × `<IMG_CONTEXT>` and
    /// `</img>`: the query is tokenized with the two ends only and the picture's tokens go between
    /// them, which gives the same ids, as special tokens split the text around them.
    private func withPicture(_ query: String, tokens: Int) -> [Int] {
        guard let range = query.range(of: "<image>") else { return tokenizer.encode(query, addSpecialTokens: true) }
        var ids = tokenizer.encode(query.replacingCharacters(in: range, with: "<img></img>"), addSpecialTokens: true)
        if let index = ids.indices.dropLast().first(where: { ids[$0] == imageStart && ids[$0 + 1] == imageEnd }) {
            ids.insert(contentsOf: Array(repeating: imageContext, count: tokens), at: index + 1)
        }
        return ids
    }
}

/// The token embeddings, keyed as the checkpoint's `embed_tokens` (read by the understanding side only).
public final class SenseNovaTokenEmbedding: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    init(config: SenseNovaConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        super.init()
    }
}

/// SenseNova-U1.5 (SenseTime's NEO-unify 8B-MoT): the prompt read once by the understanding stack
/// into a key–value cache, then the image denoised in pixel space by the generation stack, as the
/// reference's `t2i_generate` runs it: Euler steps on a shifted schedule from t = 0 (noise) to
/// t = 1, the model predicting the clean image and the velocity derived from it, activations in
/// the checkpoint's dtype (bf16) as the reference keeps them. Guidance above 1 adds the
/// unconditional pass (the MLX packs of the 8-step distillation run without it). With a picture,
/// the image is an edit of it (`it2i_generate`): the picture's tokens join the prompt, read by the
/// understanding stack's own patch embedding.
///
/// Checkpoints are the MLX packs (`mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit` and friends):
/// the original's keys and `config.json` in one folder, linears quantized, convolutions channels
/// last. Both stacks load lazily; with Save memory the understanding stack leaves memory once the
/// prompts are read, and the generation stack before a new prompt is read.
public final class SenseNovaModel: FamilyModel {
    public let config: SenseNovaConfig
    public let modelPath: URL
    public var bits: Int? { config.bits }
    public var lowRam = false
    /// The schedule's shift and the smallest 1 − t the velocity is divided by (the reference's
    /// defaults: `--timestep_shift 3`, `t_eps` 0.02).
    public var timestepShift: Double = 3
    public var tEps: Float = 0.02

    /// A query's keys and values, layer by layer, and the text position the image's tokens take
    /// after it (its last position plus one: its length for text alone).
    public struct Prefix {
        public let cache: [(MLXArray, MLXArray)]
        public let position: Int

        public init(cache: [(MLXArray, MLXArray)], position: Int) {
            self.cache = cache
            self.position = position
        }
    }

    private var prompter: SenseNovaPrompter?
    private var understanding: (embed: SenseNovaTokenEmbedding, vision: SenseNovaVisionModel, stack: SenseNovaStack)?
    private var generation: (stack: SenseNovaStack, modules: SenseNovaFlowModules)?
    /// The generation stack has run: its weights are resident rather than lazy.
    private var generationUsed = false
    private var promptCache: [String: Prefix] = [:]
    private var unconditional: Prefix?
    /// The last few edits' queries, by prompt, picture, size and guidance.
    private var editCache: [(key: String, prefix: Prefix, unconditional: Prefix?)] = []
    private static let cachedEdits = 3

    public init(modelPath: URL, loadTokenizer: Bool = true) throws {
        self.modelPath = modelPath
        config = try SenseNovaConfig.read(folder: modelPath)
        try loadGeneration()
        try loadUnderstanding()
        if loadTokenizer { prompter = try SenseNovaPrompter(folder: modelPath) }
    }

    // MARK: Weights

    /// The pack's tensors whose keys pass `include`, from every shard (lazily: only what is used is read).
    private func tensors(_ include: (String) -> Bool) throws -> [String: MLXArray] {
        let shards = try Checkpoint.shards(in: modelPath, component: "model")
        var tensors: [String: MLXArray] = [:]
        for shard in shards {
            for (key, value) in try loadArrays(url: shard) where include(key) { tensors[key] = value }
        }
        return tensors
    }

    /// The part of the tensors under `prefix`, keyed without it.
    private static func strip(_ tensors: [String: MLXArray], _ prefix: String) -> [String: MLXArray] {
        var stripped: [String: MLXArray] = [:]
        for (key, value) in tensors where key.hasPrefix(prefix) { stripped[String(key.dropFirst(prefix.count))] = value }
        return stripped
    }

    private func loadUnderstanding() throws {
        let loaded = try tensors { $0.hasPrefix("language_model.model.") && !$0.contains("_mot_gen") || $0.hasPrefix("vision_model.") }
        let embed = SenseNovaTokenEmbedding(config: config)
        try WeightLoading.apply(Self.strip(loaded, "language_model.model.").filter { $0.key.hasPrefix("embed_tokens.") }, to: embed)
        let vision = SenseNovaVisionModel(config: config)
        try WeightLoading.apply(Self.strip(loaded, "vision_model."), to: vision)
        let stack = SenseNovaStack(config: config)
        try WeightLoading.apply(SenseNovaStack.split(Self.strip(loaded, "language_model.model.")).understanding, to: stack)
        understanding = (embed, vision, stack)
    }

    private func loadGeneration() throws {
        let loaded = try tensors { $0.hasPrefix("fm_modules.") || $0.hasPrefix("language_model.model.") && $0.contains("_mot_gen") }
        let stack = SenseNovaStack(config: config)
        try WeightLoading.apply(SenseNovaStack.split(Self.strip(loaded, "language_model.model.")).generation, to: stack)
        let modules = SenseNovaFlowModules(config: config)
        try WeightLoading.apply(Self.strip(loaded, "fm_modules."), to: modules)
        generation = (stack, modules)
        generationUsed = false
    }

    /// The understanding stack, the token embeddings and the pictures' patch embedding, reloaded if
    /// they had been released; with Save memory a generation stack that has run is released first,
    /// so the two are never both resident.
    func loadedUnderstanding() throws -> (embed: SenseNovaTokenEmbedding, vision: SenseNovaVisionModel, stack: SenseNovaStack) {
        if let understanding { return understanding }
        if lowRam, generationUsed {
            generation = nil
            Memory.clearCache()
        }
        try loadUnderstanding()
        return understanding!
    }

    func loadedGeneration() throws -> (stack: SenseNovaStack, modules: SenseNovaFlowModules) {
        if let generation { return generation }
        try loadGeneration()
        return generation!
    }

    /// Replaces the quantized linears of both stacks with bf16 ones holding their dequantized
    /// weights, rounded as the reference has them when it runs a pack
    /// (Engine/Reference/sensenova_reference.py): for parity checks only, it takes twice the memory.
    public func dequantizeForParity() throws {
        for stack in [try loadedUnderstanding().stack, try loadedGeneration().stack] {
            let leaves = stack.leafModules().flattened()
            let replacements = leaves.compactMap { path, module -> (String, Module)? in
                guard let quantized = module as? QuantizedLinear else { return nil }
                let weight = dequantized(quantized.weight, scales: quantized.scales, biases: quantized.biases,
                                         groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
                return (path, Linear(weight: weight, bias: quantized.bias))
            }
            let layers = Dictionary(leaves, uniquingKeysWith: { first, _ in first })
            let tree = WeightLoading.completingLists(NestedItem.unflattened(replacements), at: "", layers: layers)
            try stack.update(modules: NestedDictionary(item: tree), verify: .none)
        }
    }

    /// The activations' dtype, the checkpoint's: bf16 in the packs, float32 in a fixture.
    public func activationType() throws -> DType { try loadedGeneration().stack.norm.weight.dtype }

    // MARK: Prompts

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil && unconditional != nil }

    /// Reads the prompt, and the unconditional query once, into their caches.
    public func encode(_ prompt: String) throws {
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        if promptCache[prompt] == nil { promptCache[prompt] = try prefix(ids: prompter.tokenIds(prompt)) }
        if unconditional == nil { unconditional = try prefix(ids: prompter.unconditionalIds) }
    }

    public func promptsEncoded() {
        if lowRam, understanding != nil {
            understanding = nil
            Memory.clearCache()
        }
    }

    /// Text tokens through the understanding stack: the keys and values the image attends to.
    public func prefix(ids: [Int]) throws -> Prefix {
        let (embed, _, stack) = try loadedUnderstanding()
        let embeds = embed.embedTokens(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        let positions = SenseNovaPositions.text(count: ids.count)
        let cache = stack.prefixCache(embeds, positions: positions, mask: SenseNovaStack.blockCausalMask(positions, dtype: embeds.dtype))
        eval(cache.flatMap { [$0.0, $0.1] })
        return Prefix(cache: cache, position: ids.count)
    }

    /// The patch embedding of the pictures an edit reads (`extract_feature`), from their normalized
    /// pixels [1, H, W, 3]: [1, tokens, hidden]. The pixels are rounded to bf16 first, as
    /// `it2i_generate` casts them whatever the model's dtype.
    public func pictureFeatures(_ pixelValues: MLXArray) throws -> MLXArray {
        let (_, vision, _) = try loadedUnderstanding()
        return vision(pixelValues.asType(.bfloat16).asType(try activationType()))
    }

    /// An edit's query through the understanding stack (`_it2i_prefix_forward`): the tokens of
    /// `ids`, the pictures' features in place of their `imageContext` tokens (in order), at the
    /// positions of `get_thw_indexes`, under the block-causal mask (a picture's tokens see each
    /// other).
    public func prefix(ids: [Int], pictures: [MLXArray], imageStart: Int, imageContext: Int) throws -> Prefix {
        let (embed, _, stack) = try loadedUnderstanding()
        let tokens = embed.embedTokens(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        var pieces: [MLXArray] = []
        var grids: [(rows: Int, columns: Int)] = []
        var start = 0
        var picture = 0
        var index = 0
        while index < ids.count {
            guard ids[index] == imageContext, picture < pictures.count else { index += 1; continue }
            let features = try pictureFeatures(pictures[picture])
            let (rows, columns) = (pictures[picture].shape[1] / config.tokenSize, pictures[picture].shape[2] / config.tokenSize)
            pieces.append(tokens[0..., start ..< index])
            pieces.append(features.asType(tokens.dtype))
            grids.append((rows, columns))
            index += rows * columns
            start = index
            picture += 1
        }
        pieces.append(tokens[0..., start...])
        let embeds = concatenated(pieces, axis: 1)
        let positions = SenseNovaPositions.query(ids: ids, imageStart: imageStart, imageContext: imageContext, grids: grids)
        let cache = stack.prefixCache(embeds, positions: positions, mask: SenseNovaStack.blockCausalMask(positions, dtype: embeds.dtype))
        eval(cache.flatMap { [$0.0, $0.1] })
        return Prefix(cache: cache, position: Int(positions.t.max() ?? -1) + 1)
    }

    /// An edit's queries (and, with guidance, the picture-only one), from the cache when the same
    /// prompt, picture and size came last.
    private func editPrefixes(prompt: String, picture: SenseNovaPicture, key: String, guided: Bool) throws -> (Prefix, Prefix?) {
        let cacheKey = "\(key)|\(picture.width)x\(picture.height)|\(guided)|\(prompt)"
        if let cached = editCache.first(where: { $0.key == cacheKey }) { return (cached.prefix, cached.unconditional) }
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        let pixels = picture.pixelValues
        let tokens = picture.tokens(tokenSize: config.tokenSize)
        func run(_ ids: [Int]) throws -> Prefix {
            try self.prefix(ids: ids, pictures: [pixels], imageStart: prompter.imageStart, imageContext: prompter.imageContext)
        }
        let prefix = try run(prompter.editIds(prompt, pictureTokens: tokens))
        let unconditional = guided ? try run(prompter.editUnconditionalIds(pictureTokens: tokens)) : nil
        editCache.append((cacheKey, prefix, unconditional))
        if editCache.count > Self.cachedEdits { editCache.removeFirst() }
        return (prefix, unconditional)
    }

    // MARK: Sampling

    /// What the model reads for the image at time `t`: its generation embedding plus the time's
    /// and the noise scale's embeddings, [1, N, hidden].
    public func imageEmbeds(_ image: MLXArray, t: Float, noiseScale: Double) throws -> MLXArray {
        let modules = try loadedGeneration().modules
        var time = modules.timestepEmbedder(t)
        if let noiseEmbedder = modules.noiseScaleEmbedder {
            time = time + noiseEmbedder(Float(noiseScale / config.noiseScaleMaxValue))
        }
        return modules.visionModel(image) + time
    }

    /// `_t2i_predict_v`: the generation stack over the image's embeddings with a prompt's cache,
    /// the pixel head's clean image, and the velocity towards it from `image`: [1, H, W, 3].
    public func velocity(embeds: MLXArray, image: MLXArray, prefix: Prefix, t: Float) throws -> MLXArray {
        let (stack, modules) = try loadedGeneration()
        let (rows, columns) = (image.shape[1] / config.tokenSize, image.shape[2] / config.tokenSize)
        let positions = SenseNovaPositions.image(rows: rows, columns: columns, t: prefix.position)
        let hidden = stack(embeds, positions: positions, prefix: prefix.cache)
        let predicted = modules.head(hidden.reshaped([1, rows, columns, config.hiddenSize]))
        // (x_pred − z) in the activations' dtype, divided by the float32 max(1 − t, t_eps).
        return ((predicted - image).asType(.float32) / Swift.max(1 - t, tEps)).asType(image.dtype)
    }

    /// The denoising loop from `noise` (unit normal, [1, H, W, 3]): scaled for the image's size,
    /// then one Euler step per pair of timesteps, with guidance when it is above 1.
    public func denoise(
        noise: MLXArray, prefix: Prefix, unconditional: Prefix?, steps: Int, guidance: Float,
        progress: (Int, Int) -> Void = { _, _ in }, isCancelled: () -> Bool = { false }
    ) throws -> MLXArray {
        let tokens = (noise.shape[1] / config.tokenSize) * (noise.shape[2] / config.tokenSize)
        let noiseScale = config.noiseScale(tokens: tokens)
        var image = noise.asType(try activationType()) * Float(noiseScale)
        let timesteps = SenseNovaConfig.timesteps(steps: steps, shift: timestepShift)
        for index in 0 ..< steps {
            let (t, next) = (timesteps[index], timesteps[index + 1])
            let embeds = try imageEmbeds(image, t: t, noiseScale: noiseScale)
            var velocity = try velocity(embeds: embeds, image: image, prefix: prefix, t: t)
            if guidance > 1, let unconditional {
                let other = try self.velocity(embeds: embeds, image: image, prefix: unconditional, t: t)
                velocity = other + ((velocity - other).asType(.float32) * guidance).asType(velocity.dtype)
            }
            image = image + (velocity.asType(.float32) * (next - t)).asType(image.dtype)
            eval(image)
            generationUsed = true
            progress(index + 1, steps)
            if isCancelled() { throw GenerationError.cancelled }
        }
        return image
    }

    /// [1, H, W, 3] in [−1, 1] → [H, W, 3] uint8, as the reference's `_to_pil` (float32, clamped,
    /// rounded).
    public static func pixels(_ image: MLXArray) -> MLXArray {
        let unit = clip(image.asType(.float32) * 0.5 + 0.5, min: 0, max: 1)
        return (unit * 255).round().asType(.uint8)[0]
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        let size = config.tokenSize
        let width = size * (request.width / size)
        let height = size * (request.height / size)
        guard width >= size, height >= size else { throw GenerationError.sizeTooSmall }

        phase(.encoding)
        let prefix: Prefix
        let other: Prefix?
        if let path = request.imagePath {
            // An edit reads its picture with the prompt, so at the start of each image; the picture
            // takes about the image's area.
            let picture = SenseNovaPicture(try QwenEditPicture(path: path), area: width * height, factor: size)
            (prefix, other) = try editPrefixes(prompt: request.prompt, picture: picture, key: QwenImageEditModel.pictureKey(path),
                                               guided: request.guidance > 1)
            promptsEncoded()
        } else {
            if !isCached(request.prompt) { try encode(request.prompt) }
            guard let cached = promptCache[request.prompt] else { throw GenerationError.cancelled }
            (prefix, other) = (cached, unconditional)
        }
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        let noise = MLXRandom.normal([1, height, width, 3], key: MLXRandom.key(UInt64(truncatingIfNeeded: request.seed)))
        let image = try denoise(
            noise: noise, prefix: prefix, unconditional: other, steps: request.steps,
            guidance: Float(request.guidance), progress: progress, isCancelled: isCancelled
        )

        phase(.decoding)
        let pixels = Self.pixels(image)
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }
}
