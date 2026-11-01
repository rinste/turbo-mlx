import Foundation
import MLX
import MLXNN

// The picture's two encoders as Idol reads them (idol-sdk's `conditioning.py`): DINOv3 ViT-H+/16's tokens after its
// own affine final norm and then a non-affine LayerNorm, and a SAMPLE of the FLUX.2 VAE's posterior, patchified,
// normalized by its batch-norm statistics, behind five zero tokens. Both in bfloat16 as the reference runs them.

/// The DINOv3 configuration the pack's `dinov3/config.json` gives.
public struct IdolDINOv3Config: Equatable, Sendable {
    public var hiddenSize = 1280
    public var layers = 32
    public var heads = 20
    public var intermediateSize = 5120
    public var patchSize = 16
    public var registers = 4
    public var ropeTheta: Float = 100
    public var layerNormEps: Float = 1e-5
    public var queryBias = true
    public var keyBias = false
    public var valueBias = true

    static func read(folder: URL) throws -> IdolDINOv3Config {
        let url = folder.appending(path: "config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw IdolConfig.ConfigError.unreadable(url) }
        func int(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
        if (json["use_gated_mlp"] as? Bool) == false { throw IdolConfig.ConfigError.unsupported("DINOv3 without its gated MLP") }
        if let act = json["hidden_act"] as? String, act != "silu" { throw IdolConfig.ConfigError.unsupported("DINOv3 with \(act)") }
        var config = IdolDINOv3Config()
        config.hiddenSize = int("hidden_size") ?? config.hiddenSize
        config.layers = int("num_hidden_layers") ?? config.layers
        config.heads = int("num_attention_heads") ?? config.heads
        config.intermediateSize = int("intermediate_size") ?? config.intermediateSize
        config.patchSize = int("patch_size") ?? config.patchSize
        config.registers = int("num_register_tokens") ?? config.registers
        config.ropeTheta = (json["rope_theta"] as? NSNumber)?.floatValue ?? config.ropeTheta
        config.layerNormEps = (json["layer_norm_eps"] as? NSNumber)?.floatValue ?? config.layerNormEps
        config.queryBias = json["query_bias"] as? Bool ?? config.queryBias
        config.keyBias = json["key_bias"] as? Bool ?? config.keyBias
        config.valueBias = json["value_bias"] as? Bool ?? config.valueBias
        return config
    }
}

final class IdolDINOv3Embeddings: Module {
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "mask_token") var maskToken: MLXArray
    @ParameterInfo(key: "register_tokens") var registerTokens: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: Conv2d

    init(_ config: IdolDINOv3Config) {
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _maskToken.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _registerTokens.wrappedValue = MLXArray.zeros([1, config.registers, config.hiddenSize])
        _patchEmbeddings.wrappedValue = Conv2d(inputChannels: 3, outputChannels: config.hiddenSize,
                                                kernelSize: .init(config.patchSize), stride: .init(config.patchSize))
        super.init()
    }

    /// Pixels [1, H, W, 3] → [1, 1 + registers + patches, hidden].
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(pixels)
        let flat = patches.reshaped([patches.dim(0), -1, patches.dim(3)])
        return concatenated([clsToken.asType(flat.dtype), registerTokens.asType(flat.dtype), flat], axis: 1)
    }
}

final class IdolDINOv3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let heads: Int

    init(_ config: IdolDINOv3Config) {
        heads = config.heads
        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, dim, bias: config.queryBias)
        _kProj.wrappedValue = Linear(dim, dim, bias: config.keyBias)
        _vProj.wrappedValue = Linear(dim, dim, bias: config.valueBias)
        _oProj.wrappedValue = Linear(dim, dim)
        super.init()
    }

    /// `cos`, `sin` [patches, headDim] in the activations' type: the rotation of the patch tokens only, by halves.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = dim / self.heads
        func split(_ y: MLXArray) -> MLXArray { y.reshaped([batch, length, self.heads, headDim]).transposed(0, 2, 1, 3) }
        let prefix = length - cos.dim(0)
        func rotate(_ y: MLXArray) -> MLXArray {
            let patches = y[0..., 0..., prefix...]
            let halves = patches.split(parts: 2, axis: -1)
            let rotated = concatenated([-halves[1], halves[0]], axis: -1)
            return concatenated([y[0..., 0..., 0 ..< prefix], patches * cos + rotated * sin], axis: 2)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: rotate(split(qProj(x))), keys: rotate(split(kProj(x))), values: split(vProj(x)),
            scale: 1 / Float(headDim).squareRoot(), mask: nil
        )
        return oProj(attended.transposed(0, 2, 1, 3).reshaped([batch, length, dim]))
    }
}

final class IdolDINOv3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: IdolDINOv3Config) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class IdolLayerScale: Module {
    @ParameterInfo(key: "lambda1") var lambda1: MLXArray

    init(_ dim: Int) {
        _lambda1.wrappedValue = MLXArray.ones([dim])
        super.init()
    }
}

final class IdolDINOv3Layer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: IdolDINOv3Attention
    @ModuleInfo(key: "layer_scale1") var layerScale1: IdolLayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: IdolDINOv3MLP
    @ModuleInfo(key: "layer_scale2") var layerScale2: IdolLayerScale

    init(_ config: IdolDINOv3Config) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attention.wrappedValue = IdolDINOv3Attention(config)
        _layerScale1.wrappedValue = IdolLayerScale(config.hiddenSize)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = IdolDINOv3MLP(config)
        _layerScale2.wrappedValue = IdolLayerScale(config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let x = x + attention(norm1(x), cos: cos, sin: sin) * layerScale1.lambda1
        return x + mlp(norm2(x)) * layerScale2.lambda1
    }
}

final class IdolDINOv3Encoder: Module {
    @ModuleInfo(key: "layer") var layers: [IdolDINOv3Layer]

    init(_ config: IdolDINOv3Config) {
        _layers.wrappedValue = (0 ..< config.layers).map { _ in IdolDINOv3Layer(config) }
        super.init()
    }
}

/// transformers' `DINOv3ViTModel` (DINOv3 ViT-H+/16), its tensors as transformers 4.56 names them.
public final class IdolDINOv3: Module {
    @ModuleInfo(key: "embeddings") var embeddings: IdolDINOv3Embeddings
    @ModuleInfo(key: "model") var model: IdolDINOv3Encoder
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let config: IdolDINOv3Config
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    init(_ config: IdolDINOv3Config) {
        self.config = config
        _embeddings.wrappedValue = IdolDINOv3Embeddings(config)
        _model.wrappedValue = IdolDINOv3Encoder(config)
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    /// The pack's `dinov3/`: its blocks under `layer.N` become `model.layer.N`, the patch embedding is turned
    /// channels-last, and everything is held in bfloat16, as the reference loads it.
    public static func load(folder: URL) throws -> IdolDINOv3 {
        let model = IdolDINOv3(try IdolDINOv3Config.read(folder: folder))
        var tensors: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: folder.appending(path: "model.safetensors")) {
            var value = value.asType(.bfloat16)
            if key == "embeddings.patch_embeddings.weight" { value = value.transposed(0, 2, 3, 1) }
            tensors[key.hasPrefix("layer.") ? "model.\(key)" : key] = value
        }
        try WeightLoading.apply(tensors, to: model)
        eval(model)
        return model
    }

    /// `DINOv3ViTRopePositionEmbedding` for a grid of `rows` × `columns` patches: their centres in [-1, 1], (y, x)
    /// angles at 16 frequencies each, tiled twice, as cos and sin [patches, headDim] in float32.
    func rope(rows: Int, columns: Int) -> (MLXArray, MLXArray) {
        let headDim = config.hiddenSize / config.heads
        let count = headDim / 4
        let inverse = (0 ..< count).map { 1 / pow(config.ropeTheta, Float(4 * $0) / Float(headDim)) }
        var angles = [Float]()
        angles.reserveCapacity(rows * columns * headDim)
        for y in 0 ..< rows {
            let cy = 2 * ((Float(y) + 0.5) / Float(rows)) - 1
            for x in 0 ..< columns {
                let cx = 2 * ((Float(x) + 0.5) / Float(columns)) - 1
                var row = inverse.map { 2 * Float.pi * cy * $0 } + inverse.map { 2 * Float.pi * cx * $0 }
                row += row
                angles += row
            }
        }
        let table = MLXArray(angles).reshaped([rows * columns, headDim])
        return (MLX.cos(table), MLX.sin(table))
    }

    /// The prepared picture [1, 1024, 1024, 3] in [0, 1] (float32) → Idol's `feature1` [1, 4101, 1280], float32.
    /// `inspect` sees the embeddings, every layer's output (`layer<N>`) and the last hidden state.
    public func tokens(_ pixels: MLXArray, inspect: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let mean = MLXArray(Self.mean), std = MLXArray(Self.std)
        let input = ((pixels.asType(.float32) - mean) / std).asType(.bfloat16)
        var h = embeddings(input)
        inspect?("embeddings", h)
        let (cos, sin) = rope(rows: pixels.dim(1) / config.patchSize, columns: pixels.dim(2) / config.patchSize)
        let (c, s) = (cos.asType(.bfloat16), sin.asType(.bfloat16))
        for (index, layer) in model.layers.enumerated() {
            h = layer(h, cos: c, sin: s)
            inspect?("layer\(index)", h)
        }
        let hidden = norm(h)
        inspect?("hidden", hidden)
        return MLXFast.layerNorm(hidden.asType(.float32), weight: nil, bias: nil, eps: 1e-5)
    }
}

/// The FLUX.2 VAE's encoder alone (the pack's `vae/`: channels-last convolutions, the middle attention 8-bit), and
/// what Idol makes of it: `feature2`.
public final class IdolVAEEncoder: Module {
    @ModuleInfo(key: "encoder") var encoder: Flux2Encoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats

    override init() {
        _encoder.wrappedValue = Flux2Encoder()
        _quantConv.wrappedValue = Conv2d(inputChannels: 2 * Flux2VAE.latentChannels, outputChannels: 2 * Flux2VAE.latentChannels,
                                         kernelSize: 1, stride: 1, padding: 0)
        _bn.wrappedValue = Flux2BatchNormStats(features: 4 * Flux2VAE.latentChannels)
        super.init()
    }

    public static func load(folder: URL) throws -> IdolVAEEncoder {
        let model = IdolVAEEncoder()
        try WeightLoading.apply(try loadArrays(url: folder.appending(path: "model.safetensors")), to: model)
        eval(model)
        return model
    }

    /// The posterior's moments [1, 128, 128, 64] in bfloat16: the mean, then the log-variance.
    public func moments(_ pixels: MLXArray) -> MLXArray {
        quantConv(encoder(pixels.asType(.bfloat16) * 2 - 1))
    }

    /// `feature2` [1, 4101, 128] float32 from the moments and the posterior's noise `eps` [1, 128, 128, 32]
    /// (bfloat16): a sample, patchified, normalized, behind five zero tokens.
    public func tokens(moments: MLXArray, eps: MLXArray) -> MLXArray {
        let parts = moments.split(parts: 2, axis: -1)
        let logvar = clip(parts[1], min: -30, max: 20)
        let std = exp(0.5 * logvar)
        let sample = parts[0] + std * eps.asType(moments.dtype)
        let patched = Flux2VAE.patchify(sample)
        let mean = bn.runningMean.asType(patched.dtype)
        let deviation = sqrt(bn.runningVar + bn.eps).asType(patched.dtype)
        let normalized = ((patched - mean) / deviation).asType(.float32)
        let tokens = normalized.reshaped([1, -1, normalized.dim(3)])
        return concatenated([MLXArray.zeros([1, 5, tokens.dim(2)]), tokens], axis: 1)
    }

    /// The posterior's noise for a seed, drawn as MLX draws it.
    public static func noise(shape: [Int], key: MLXArray) -> MLXArray {
        MLXRandom.normal(shape, dtype: .bfloat16, key: key)
    }
}
