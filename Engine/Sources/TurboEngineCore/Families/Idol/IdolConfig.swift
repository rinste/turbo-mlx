import Foundation

/// The geometry of an Idol checkpoint (speridlabs/idol, Apache-2.0: one picture of one object to 3D Gaussians), read
/// from the `config.json` next to its weights: the denoiser's arguments as idol-sdk's `LatentSeqMMFlowModel` takes
/// them, the size the encoders see the picture at, the octree's depth. The MLX pack (`ddalcu/Idol-MLX-Serve-8bit`)
/// keeps the release's file, with `model_type: idol` added.
public struct IdolConfig: Equatable, Sendable {
    public struct Denoiser: Equatable, Sendable {
        /// The object's latent: `tokens` × `latentChannels` (`q_token_length`, `in_channels`).
        public var tokens = 8192
        public var latentChannels = 16
        /// The input camera, one token: a direction, an inverse distance, a half width.
        public var cameraChannels = 5
        /// `model_channels`.
        public var dim = 1024
        /// The picture's tokens: DINOv3's (`cond_channels`) and the FLUX.2 VAE's (`cond2_channels`).
        public var condChannels = 1280
        public var cond2Channels = 128
        public var blocks = 24
        public var refinerBlocks = 2
        public var heads = 16
        /// `num_head_channels`, which the release's config leaves at its default.
        public var headDim = 64
        public var mlpRatio = 4.0

        public var mlpDim: Int { Int(Double(dim) * mlpRatio) }
    }

    /// The octree decoder (`OctreeProbabilityFixedlenDecoder`): cross attention to the latent's 16 channels.
    public struct Decoder: Equatable, Sendable {
        public var dim = 1024
        public var contextChannels = 16
        public var blocks = 4
        public var heads = 16
        public var mlpRatio = 4.0
        public var mlpDim: Int { Int(Double(dim) * mlpRatio) }
    }

    /// The Gaussian decoder (`GaussianFixedlenDecoder`) and its `representation_config`.
    public struct GaussianDecoder: Equatable, Sendable {
        public struct LearningRates: Equatable, Sendable {
            public var xyz: Float = 1
            public var colour: Float = 1
            public var opacity: Float = 1
            public var scaling: Float = 1
            public var rotation: Float = 0.1
        }
        public var dim = 1024
        public var contextChannels = 16
        public var blocks = 16
        public var heads = 16
        public var mlpRatio = 4.0
        /// Gaussians per anchor.
        public var perAnchor = 32
        public var learningRates = LearningRates()
        public var perturbSize: Float = 1.5
        public var minimumKernelSize: Float = 0.0009
        public var scalingBias: Float = 0.004
        public var opacityBias: Float = 0.1
        public var mlpDim: Int { Int(Double(dim) * mlpRatio) }
    }

    public var denoiser = Denoiser()
    public var decoder = Decoder()
    public var gaussianDecoder = GaussianDecoder()
    public var imageSize = 1024
    public var maxVoxelLevel = 8

    public init() {}

    public enum ConfigError: LocalizedError {
        case unreadable(URL)
        case unsupported(String)

        public var errorDescription: String? {
            switch self {
            case .unreadable(let url): "Couldn’t read the model’s configuration at \(url.path)."
            case .unsupported(let what): "This Idol checkpoint uses \(what), which the engine does not run."
            }
        }
    }

    /// Reads `config.json` in `folder`.
    public static func read(folder: URL) throws -> IdolConfig {
        let url = folder.appending(path: "config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let denoiser = json["denoiser"] as? [String: Any],
              let args = denoiser["args"] as? [String: Any]
        else { throw ConfigError.unreadable(url) }
        if let name = denoiser["name"] as? String, name != "LatentSeqMMFlowModel" {
            throw ConfigError.unsupported("the denoiser \(name)")
        }
        func int(_ object: [String: Any], _ key: String) -> Int? { (object[key] as? NSNumber)?.intValue }

        var config = IdolConfig()
        var d = config.denoiser
        d.tokens = int(args, "q_token_length") ?? d.tokens
        d.latentChannels = int(args, "in_channels") ?? d.latentChannels
        d.cameraChannels = int(args, "cam_channels") ?? d.cameraChannels
        d.dim = int(args, "model_channels") ?? d.dim
        d.condChannels = int(args, "cond_channels") ?? d.condChannels
        d.cond2Channels = int(args, "cond2_channels") ?? d.cond2Channels
        d.blocks = int(args, "num_blocks") ?? d.blocks
        d.refinerBlocks = int(args, "num_refiner_blocks") ?? d.refinerBlocks
        d.heads = int(args, "num_heads") ?? d.heads
        d.headDim = int(args, "num_head_channels") ?? d.headDim
        d.mlpRatio = (args["mlp_ratio"] as? NSNumber)?.doubleValue ?? d.mlpRatio
        guard d.heads * d.headDim == d.dim else {
            throw ConfigError.unsupported("\(d.heads) heads of \(d.headDim) channels in a width of \(d.dim)")
        }
        config.denoiser = d
        func float(_ object: [String: Any], _ key: String) -> Float? { (object[key] as? NSNumber)?.floatValue }
        if let decoder = (json["decoder"] as? [String: Any])?["args"] as? [String: Any] {
            config.decoder.dim = int(decoder, "model_channels") ?? config.decoder.dim
            config.decoder.contextChannels = int(decoder, "cond_channels") ?? config.decoder.contextChannels
            config.decoder.blocks = int(decoder, "num_blocks") ?? config.decoder.blocks
            config.decoder.heads = int(decoder, "num_heads") ?? config.decoder.heads
            config.decoder.mlpRatio = (decoder["mlp_ratio"] as? NSNumber)?.doubleValue ?? config.decoder.mlpRatio
        }
        if let gs = (json["decoder_gs"] as? [String: Any])?["args"] as? [String: Any] {
            var g = config.gaussianDecoder
            g.dim = int(gs, "model_channels") ?? g.dim
            g.contextChannels = int(gs, "cond_channels") ?? g.contextChannels
            g.blocks = int(gs, "num_blocks") ?? g.blocks
            g.heads = int(gs, "num_heads") ?? g.heads
            g.mlpRatio = (gs["mlp_ratio"] as? NSNumber)?.doubleValue ?? g.mlpRatio
            if let rep = gs["representation_config"] as? [String: Any] {
                g.perAnchor = int(rep, "num_gaussians") ?? g.perAnchor
                g.perturbSize = float(rep, "perturbe_size") ?? g.perturbSize
                g.minimumKernelSize = float(rep, "3d_filter_kernel_size") ?? g.minimumKernelSize
                g.scalingBias = float(rep, "scaling_bias") ?? g.scalingBias
                g.opacityBias = float(rep, "opacity_bias") ?? g.opacityBias
                if let lr = rep["lr"] as? [String: Any] {
                    g.learningRates.xyz = float(lr, "_xyz") ?? g.learningRates.xyz
                    g.learningRates.colour = float(lr, "_features_dc") ?? g.learningRates.colour
                    g.learningRates.opacity = float(lr, "_opacity") ?? g.learningRates.opacity
                    g.learningRates.scaling = float(lr, "_scaling") ?? g.learningRates.scaling
                    g.learningRates.rotation = float(lr, "_rotation") ?? g.learningRates.rotation
                }
            }
            config.gaussianDecoder = g
        }
        config.imageSize = int(json, "image_size") ?? config.imageSize
        config.maxVoxelLevel = int(json, "max_voxel_level") ?? config.maxVoxelLevel
        return config
    }
}
