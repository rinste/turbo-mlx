import Foundation
import MLX
import MLXNN

/// Z-Image Turbo: the Qwen3 encoder, the S3-DiT and the decoder, with the prompt cache and the
/// sampling loop of mflux's `ZImage.generate_image` (guidance off, the linear schedule).
public final class ZImageModel: PreviewingFamilyModel, LoRAAdaptable, ImageToImageFamily {
    public let config: ZImageConfig
    public let modelPath: URL
    public let textEncoder: Qwen3TextEncoder
    public let transformer: S3DiTTransformer
    public let vae: ZImageVAE
    public private(set) var bits: Int?
    /// Save memory: decode in tiles. The Qwen3 4B encoder is small enough to stay resident.
    public var lowRam = false
    private var prompter: Qwen3Prompter?
    /// Caption features per prompt: [tokens, capFeatDim] bf16.
    private var promptCache: [String: MLXArray] = [:]
    public let loras = LoRAAdapters(table: .zImage)
    public var adaptedModule: Module? { transformer }
    /// The VAE's encoder, loaded the first time an image starts from a picture.
    private var pictureEncoder: ZImageEncoder?
    /// The last picture encoded, so the images of a batch encode it once.
    private var startingPicture: (key: String, latents: MLXArray)?

    /// Loads the checkpoint at `modelPath` (mflux format). Weights stay lazy until first use.
    public init(modelPath: URL, config: ZImageConfig, loadTokenizer: Bool = true) throws {
        self.config = config
        self.modelPath = modelPath
        textEncoder = Qwen3TextEncoder(config: config.textEncoder)
        transformer = S3DiTTransformer(config: config.transformer)
        vae = ZImageVAE(blockOutChannels: config.vaeBlockOutChannels)

        var checkpoint = Checkpoint(root: modelPath)
        try WeightLoading.apply(try checkpoint.loadComponent("text_encoder"), to: textEncoder, ignoring: Qwen3TextEncoder.ignoresKey)
        try WeightLoading.apply(try checkpoint.loadComponent("transformer"), to: transformer)
        try WeightLoading.apply(try checkpoint.loadComponent("vae"), to: vae, ignoring: ZImageVAE.ignoresKey)
        bits = checkpoint.bits
        if loadTokenizer {
            prompter = try Qwen3Prompter(modelPath: modelPath, maxLength: config.maxSequenceLength, enableThinking: true)
        }
    }

    public func isCached(_ prompt: String) -> Bool { promptCache[prompt] != nil }

    public func encode(_ prompt: String) throws {
        _ = try encodePrompt(prompt)
    }

    public func promptsEncoded() {}

    /// The caption features of a prompt (once; later calls read the cache).
    @discardableResult
    public func encodePrompt(_ prompt: String) throws -> MLXArray {
        if let cached = promptCache[prompt] { return cached }
        guard let prompter else { throw Qwen3Prompter.PromptError.noTokenizer(modelPath) }
        // The encoder is causal and the padding is masked, so the real tokens' states are the
        // same without the padding mflux adds to 512; only they are read (`cap_feats[0, :num_valid]`).
        let ids = try prompter.tokenIds(prompt)
        let inputIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let encoded = encode(inputIds: inputIds)
        promptCache[prompt] = encoded
        return encoded
    }

    /// The second-to-last hidden state of the Qwen3 encoder run in float32, as bf16: [tokens, capFeatDim].
    public func encode(inputIds: MLXArray) -> MLXArray {
        let mask = MLXArray.ones(inputIds.shape, dtype: .int32)
        let layer = config.textEncoder.numHiddenLayers - 1
        let states = textEncoder.hiddenStates(inputIds: inputIds, attentionMask: mask, computeType: .float32, through: layer)
        let features = states[layer].asType(modelPrecision)[0]
        eval(features)
        return features
    }

    // MARK: Sampling

    /// The initial noise for a size and seed: [16, 1, H/8, W/8] in the model's precision.
    public static func initialLatents(width: Int, height: Int, seed: Int) -> MLXArray {
        MLXRandom.normal([ZImageVAE.latentChannels, 1, height / 8, width / 8], key: MLXRandom.key(UInt64(seed)))
            .asType(modelPrecision)
    }

    public func generate(
        _ request: FamilyRequest,
        phase: (GenerationPhase) -> Void,
        progress: (Int, Int) -> Void,
        preview: (Preview) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedImage {
        let width = 16 * (request.width / 16)
        let height = 16 * (request.height / 16)
        guard width >= 16, height >= 16 else { throw GenerationError.sizeTooSmall }

        phase(.encoding)
        let capFeats = try encodePrompt(request.prompt)
        if isCancelled() { throw GenerationError.cancelled }

        // Image to image: the picture's latents, mixed with the noise at the sigma of the step the
        // loop starts from (mflux's `Img2Img`).
        let start = request.imagePath == nil ? 0 : ImageToImage.startStep(steps: request.steps, strength: request.imageStrength)
        let picture = start > 0 ? try startingLatents(path: request.imagePath!, width: width, height: height) : nil
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising)
        // The reference lets its float32 timestep embedding promote the stream to float32; the
        // 16-bit option keeps it in the weights' precision, as Ming-Image's S3-DiT always does.
        transformer.setKeepsWeightsPrecision(request.halfPrecision)
        let schedule = LinearSchedule(steps: request.steps, width: width, height: height, shift: config.shift)
        var latents = Self.initialLatents(width: width, height: height, seed: request.seed)
        if let picture { latents = ImageToImage.noised(clean: picture, noise: latents, sigma: schedule.sigmas[start]) }
        let steps = request.steps - start
        for t in start ..< request.steps {
            let sigma = schedule.sigmas[t]
            let noise = transformer(latents: latents, timestep: MLXArray([1 - sigma]), capFeats: capFeats)
            let glimpse = LatentPreview.shows(step: t - start + 1, of: steps)
                ? LatentPreview.predicted(latents: latents, noise: noise, sigma: sigma) : nil
            latents = schedule.step(latents: latents, noise: noise, index: t)
            eval(latents)
            progress(t - start + 1, steps)
            if let glimpse { preview(Preview(pixels: previewPixels(latents: glimpse), step: t - start + 1)) }
            if isCancelled() { throw GenerationError.cancelled }
        }

        phase(.decoding)
        // An image that started from a picture leaves float32 latents (mflux's mix promotes them),
        // which mflux decodes as they are; in 16 bits, like a text-to-image's, the decoder peaks as
        // it does for those (at 1024 px without Save memory, 11.7 GB with the encoder's instead of 15.9).
        if picture != nil { latents = latents.asType(modelPrecision) }
        let pixels = try decode(latents: latents, isCancelled: isCancelled)
        eval(pixels)
        return GeneratedImage(pixels: pixels)
    }

    /// The picture at `path` as the loop's clean latents, [16, 1, H/8, W/8] like the noise
    /// (`ZImageLatentCreator.pack_latents`): resized to the image's size, encoded, kept for the
    /// next image of the batch. The encoder's weights are read from the checkpoint the first time.
    public func startingLatents(path: String, width: Int, height: Int) throws -> MLXArray {
        let key = "\(path)|\(width)x\(height)"
        if let startingPicture, startingPicture.key == key { return startingPicture.latents }
        let latents = try startingLatents(pixels: try ImageToImage.pixels(path: path, width: width, height: height))
        startingPicture = (key, latents)
        return latents
    }

    /// A picture [1, H, W, 3] in [-1, 1] encoded as the loop's clean latents [16, 1, H/8, W/8], in
    /// tiles with Save memory as mflux's low-RAM mode encodes it. The encoder's weights are read from
    /// the checkpoint the first time.
    public func startingLatents(pixels: MLXArray) throws -> MLXArray {
        if pictureEncoder == nil {
            let encoder = ZImageEncoder(blockOutChannels: config.vaeBlockOutChannels)
            var checkpoint = Checkpoint(root: modelPath)
            try WeightLoading.apply(ZImageEncoder.weights(try checkpoint.loadComponent("vae")), to: encoder)
            pictureEncoder = encoder
        }
        let encoder = pictureEncoder!
        let grid = lowRam ? try VAETiling.encode(pixels, spatialScale: ZImageVAE.spatialScale, encode: { encoder.encode($0) })
            : encoder.encode(pixels)
        // [1, h, w, 16] channels-last → [16, 1, h, w].
        let latents = grid.transposed(3, 0, 1, 2)
        eval(latents)
        return latents
    }

    /// A small image of what `latents` ([16, 1, h, w], a step's prediction) hold: the grid pooled
    /// to about 384 pixels on the longer side, decoded in one piece.
    func previewPixels(latents: MLXArray) -> MLXArray {
        let grid = latents.reshaped([1, latents.shape[0], latents.shape[2], latents.shape[3]]).transposed(0, 2, 3, 1)
        let factor = LatentPreview.factor(height: grid.shape[1], width: grid.shape[2], scale: ZImageVAE.spatialScale)
        return Pixels.toPixels(vae.decode(LatentPreview.pooled(grid, factor: factor)))
    }

    /// [16, 1, h, w] latents → [H, W, 3] uint8 pixels; in tiles with Save memory on.
    public func decode(latents: MLXArray, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let grid = latents.reshaped([1, latents.shape[0], latents.shape[2], latents.shape[3]]).transposed(0, 2, 3, 1)
        let decoded = lowRam
            ? try VAETiling.decode(grid, spatialScale: ZImageVAE.spatialScale, decode: { vae.decode($0) }, isCancelled: isCancelled)
            : vae.decode(grid)
        return Pixels.toPixels(decoded)
    }
}
