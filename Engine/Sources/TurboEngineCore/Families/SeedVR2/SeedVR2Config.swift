import Foundation

/// The geometry of a SeedVR2 checkpoint: mflux's `SeedVR2Transformer` defaults (the 3B model) and
/// `SeedVR2VAE`. A fixture (see Fixtures/) shrinks every dimension for a parity check.
public struct SeedVR2Config: Equatable, Sendable {
    public var name = "seedvr2-3b"
    /// The two files of the checkpoint, as numz/SeedVR2_comfyUI names them.
    public var transformerFile = "seedvr2_ema_3b_fp16.safetensors"
    public var vaeFile = "ema_vae_fp16.safetensors"

    // Transformer.
    public var vidInChannels = 33
    public var vidOutChannels = 16
    public var vidDim = 2560
    public var txtInDim = 5120
    public var heads = 20
    public var headDim = 128
    public var expandRatio = 4
    public var normEps: Float = 1e-5
    public var numLayers = 32
    /// Blocks below this one keep separate video and text weights; the others share them.
    public var mmLayers = 10
    public var ropeDim = 128
    /// Windows per axis (time, height, width) the video tokens attend within.
    public var window = (t: 4, h: 3, w: 3)

    // VAE.
    public var latentChannels = 16
    public var blockOutChannels = [128, 256, 512, 512]
    public var scalingFactor: Float = 0.9152

    public var embDim: Int { 6 * vidDim }

    public init() {}

    public static let seedVR2_3B = SeedVR2Config()

    public static func == (a: SeedVR2Config, b: SeedVR2Config) -> Bool {
        a.name == b.name && a.vidDim == b.vidDim && a.heads == b.heads && a.headDim == b.headDim && a.numLayers == b.numLayers
            && a.mmLayers == b.mmLayers && a.ropeDim == b.ropeDim && a.window == b.window && a.txtInDim == b.txtInDim
            && a.blockOutChannels == b.blockOutChannels
    }
}
