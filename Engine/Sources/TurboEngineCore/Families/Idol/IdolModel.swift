import Foundation
import MLX
import MLXNN

/// What a family making 3D objects reports while it works, in order.
public enum MeshPhase {
    case segmenting
    case encoding
    case denoising(total: Int)
    case decoding
    case saving
}

/// One object made: the discs written to the GLB and the Gaussians they came from.
public struct GeneratedMesh {
    public let discs: Int
    public let gaussians: Int
    public let bytes: Int
}

/// A family that makes a 3D object from a picture (Idol): it writes the GLB itself, at `output`.
public protocol MeshFamilyModel: FamilyModel {
    func generateMesh(
        _ request: FamilyRequest,
        output: URL,
        phase: (MeshPhase) -> Void,
        progress: (Int, Int) -> Void,
        isCancelled: () -> Bool
    ) throws -> GeneratedMesh
}

public enum IdolError: LocalizedError {
    case objectsOnly
    case noPicture
    case gaussians(Int)
    case steps(Int)
    case mode(String)
    case missingFile(String, URL)

    public var errorDescription: String? {
        switch self {
        case .objectsOnly: "Idol makes 3D objects, not images."
        case .noPicture: "Idol makes an object from a picture: give it one."
        case .gaussians(let n): "Idol makes 32,768 to 1,048,576 Gaussians, a multiple of 32; \(n) is not."
        case .steps(let n): "Idol takes 1 to 100 steps; \(n) is not."
        case .mode(let mode): "Idol's mode is “synthetic” (renders, generated pictures) or “real” (photos), not “\(mode)”."
        case .missingFile(let name, let url): "The Idol model at \(url.path) has no \(name)."
        }
    }

    /// The request's settings, refused by name when Idol cannot run them.
    public static func validate(_ request: FamilyRequest) throws {
        guard request.imagePath != nil else { throw noPicture }
        let gaussians = request.gaussians ?? IdolModel.defaultGaussians
        guard (32768 ... 1_048_576).contains(gaussians), gaussians % 32 == 0 else { throw IdolError.gaussians(gaussians) }
        guard (1 ... 100).contains(request.steps) else { throw IdolError.steps(request.steps) }
        let mode = request.mode ?? "synthetic"
        guard IdolModel.margins[mode] != nil else { throw IdolError.mode(mode) }
    }
}

/// speridlabs' Idol (Apache-2.0) from the MLX pack (`ddalcu/Idol-MLX-Serve-8bit`): one picture of one object → 3D
/// Gaussians → the disc-mesh GLB. The picture's cutout (BiRefNet, unless it has an alpha of its own) and crop, its
/// DINOv3 and VAE tokens, the denoiser's guided Euler steps, the octree and Gaussian decoders. With Save memory,
/// BiRefNet and DINOv3 are let go once used and loaded again for the next object.
public final class IdolModel: MeshFamilyModel {
    public static let defaultGaussians = 262_144
    /// idol-sdk's `CROP_MARGIN`: the crop's side over the object's.
    static let margins: [String: Double] = ["synthetic": 1.2, "real": 1.7]

    public let folder: URL
    public let config: IdolConfig
    public let denoiser: IdolDenoiser
    public let decoder: IdolOctreeDecoder
    public let gaussianDecoder: IdolGaussianDecoder
    public let vae: IdolVAEEncoder
    private var matte: IdolMatte?
    private var dinov3: IdolDINOv3?
    public var lowRam = false
    /// For checks only (`turbo-engine bench-idol --exact`): the posterior's noise and the denoiser's starting latent
    /// and camera taken from a recording instead of drawn, so the object is the reference's own.
    public var noiseOverride: (eps: MLXArray, latent: MLXArray, camera: MLXArray)?
    public var bits: Int? { 8 }

    public init(folder: URL) throws {
        self.folder = folder
        for name in ["config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors", "birefnet.safetensors",
                     "dinov3/model.safetensors", "vae/model.safetensors"]
        where !FileManager.default.fileExists(atPath: folder.appending(path: name).path) {
            throw IdolError.missingFile(name, folder)
        }
        config = try IdolConfig.read(folder: folder)
        denoiser = try IdolDenoiser.load(folder: folder)
        decoder = try Self.loadDecoder(IdolOctreeDecoder(config: config.decoder), file: "decoder.safetensors", folder: folder)
        gaussianDecoder = try Self.loadDecoder(IdolGaussianDecoder(config: config.gaussianDecoder), file: "decoder_gs.safetensors", folder: folder)
        vae = try IdolVAEEncoder.load(folder: folder.appending(path: "vae"))
        matte = try IdolMatte.load(url: folder.appending(path: "birefnet.safetensors"))
        dinov3 = try IdolDINOv3.load(folder: folder.appending(path: "dinov3"))
    }

    public static func loadDecoder<M: Module>(_ model: M, file: String, folder: URL) throws -> M {
        var tensors: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: folder.appending(path: file)) { tensors[IdolDenoiser.renamed(key)] = value }
        try WeightLoading.apply(tensors, to: model)
        eval(model)
        return model
    }

    // No prompt: the text side of `FamilyModel` does nothing.
    public func isCached(_ prompt: String) -> Bool { true }
    public func encode(_ prompt: String) throws {}
    public func promptsEncoded() {}

    public func generate(_ request: FamilyRequest, phase: (GenerationPhase) -> Void, progress: (Int, Int) -> Void,
                         isCancelled: () -> Bool) throws -> GeneratedImage {
        throw IdolError.objectsOnly
    }

    /// The picture's preparation (cutout, crop, 1024 × 1024 on black), RGB bytes.
    public func prepare(_ picture: IdolPicture, mode: String) throws -> [UInt8] {
        let matte = try self.matte ?? IdolMatte.load(url: folder.appending(path: "birefnet.safetensors"))
        defer { self.matte = lowRam ? nil : matte }
        return try IdolPreparation.prepare(picture, size: config.imageSize, margin: Self.margins[mode] ?? 1.2) { rgb, width, height in
            matte.alpha(rgb: rgb, width: width, height: height)
        }
    }

    /// The prepared picture's tokens, DINOv3's and the VAE's: `eps` is the posterior's noise.
    public func encode(prepared: [UInt8], eps: MLXArray) throws -> (feature1: MLXArray, feature2: MLXArray) {
        let size = config.imageSize
        let pixels = MLXArray(prepared.map { Float($0) / 255 }).reshaped([1, size, size, 3])
        let dinov3 = try self.dinov3 ?? IdolDINOv3.load(folder: folder.appending(path: "dinov3"))
        let feature1 = dinov3.tokens(pixels)
        eval(feature1)
        self.dinov3 = lowRam ? nil : dinov3
        if lowRam { Memory.clearCache() }
        let feature2 = vae.tokens(moments: vae.moments(pixels), eps: eps)
        eval(feature2)
        return (feature1, feature2)
    }

    public func generateMesh(
        _ request: FamilyRequest, output: URL, phase: (MeshPhase) -> Void, progress: (Int, Int) -> Void, isCancelled: () -> Bool
    ) throws -> GeneratedMesh {
        try IdolError.validate(request)
        let mode = request.mode ?? "synthetic"
        let gaussians = request.gaussians ?? Self.defaultGaussians
        let picture = try IdolPicture.load(request.imagePath!)

        phase(.segmenting)
        let prepared = try prepare(picture, mode: mode)
        if isCancelled() { throw GenerationError.cancelled }

        // The seed's draws: the posterior's noise, then the latent's and the camera's.
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(truncatingIfNeeded: request.seed)), into: 3)
        let latentSize = config.imageSize / 8
        let eps = noiseOverride?.eps ?? IdolVAEEncoder.noise(shape: [1, latentSize, latentSize, Flux2VAE.latentChannels], key: keys[0])
        phase(.encoding)
        let (feature1, feature2) = try encode(prepared: prepared, eps: eps)
        let conditional = denoiser.context(feature1: feature1, feature2: feature2)
        let unconditional = denoiser.context(feature1: zeros(like: feature1), feature2: zeros(like: feature2))
        eval(conditional, unconditional)
        if isCancelled() { throw GenerationError.cancelled }

        phase(.denoising(total: request.steps))
        let d = config.denoiser
        let noise = noiseOverride?.latent ?? MLXRandom.normal([1, d.tokens, d.latentChannels], key: keys[1])
        let cameraNoise = noiseOverride?.camera ?? MLXRandom.normal([1, 1, d.cameraChannels], key: keys[2])
        var sampler = IdolSampler(steps: request.steps, guidance: Float(request.guidance))
        sampler.collapsesUnconditional = true
        let sample: (latent: MLXArray, camera: MLXArray)
        do {
            sample = try sampler.run(denoiser, latent: noise, camera: cameraNoise, conditional: conditional, unconditional: unconditional,
                                     progress: progress, isCancelled: isCancelled)
        } catch is CancellationError {
            throw GenerationError.cancelled
        }

        phase(.decoding)
        var random = IdolRandom(seed: request.seed)
        let anchors: MLXArray
        do {
            anchors = try decoder.sample(latent: sample.latent, count: gaussians / config.gaussianDecoder.perAnchor,
                                         levels: config.maxVoxelLevel, random: &random, isCancelled: isCancelled)
        } catch is CancellationError {
            throw GenerationError.cancelled
        }
        let decoded = gaussianDecoder.decode(points: anchors, latent: sample.latent)

        phase(.saving)
        let camera = sample.camera.reshaped([-1]).asArray(Float.self)
        let written = try IdolMesh.write(decoded, camera: Array(camera.prefix(3)), to: output)
        return GeneratedMesh(discs: written.discs, gaussians: decoded.count, bytes: written.bytes)
    }
}
