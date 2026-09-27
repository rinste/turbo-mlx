import Foundation
import MLX
import MLXNN

// Ming's `connector/` folder, after mflux's `ming_connector.py`: a Qwen2-1.5B decoder stack used as
// a bidirectional encoder over the query-token states (upstream flips every `is_causal` to false).
// No embeddings or language head; the output is the final-norm hidden state.

final class ConnectorAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let config: MingConfig.Connector

    init(config: MingConfig.Connector) {
        self.config = config
        _qProj.wrappedValue = Linear(config.hiddenSize, config.numHeads * config.headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.numKvHeads * config.headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.numKvHeads * config.headDim, bias: true)
        _oProj.wrappedValue = Linear(config.numHeads * config.headDim, config.hiddenSize, bias: false)
        super.init()
    }

    /// `cos`/`sin` [L, headDim] float32.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let (heads, kvHeads, headDim) = (config.numHeads, config.numKvHeads, config.headDim)
        var q = qProj(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([batch, length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        q = Self.rope(q, cos: cos, sin: sin)
        k = Self.rope(k, cos: cos, sin: sin)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: nil
        )
        return oProj(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDim]))
    }

    /// Rotate-half over the full head, cos and sin in x's dtype.
    static func rope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let cos = cos.asType(x.dtype)
        let sin = sin.asType(x.dtype)
        let half = x.shape[x.ndim - 1] / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, 0 ..< half]], axis: -1)
        return x * cos + rotated * sin
    }
}

final class ConnectorLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: ConnectorAttention
    @ModuleInfo(key: "mlp") var mlp: LingMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: MingConfig.Connector) {
        _selfAttn.wrappedValue = ConnectorAttention(config: config)
        _mlp.wrappedValue = LingMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediate)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let x = x + selfAttn(inputLayerNorm(x), cos: cos, sin: sin)
        return x + mlp(postAttentionLayerNorm(x))
    }
}

public final class MingConnector: Module {
    @ModuleInfo(key: "layers") var layers: [ConnectorLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public let config: MingConfig.Connector

    public init(config: MingConfig.Connector) {
        self.config = config
        _layers.wrappedValue = (0 ..< config.numLayers).map { _ in ConnectorLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsEps)
        super.init()
    }

    /// `x` [B, L, hidden] → the final-norm states, positions 0..<L.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dim = config.headDim
        let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let invFreq = 1 / pow(config.ropeTheta, exponents)
        let positions = MLXArray((0 ..< x.shape[1]).map { Float($0) }).expandedDimensions(axis: 1)
        let freqs = positions * invFreq.expandedDimensions(axis: 0)
        let emb = concatenated([freqs, freqs], axis: -1)
        let cosTable = cos(emb)
        let sinTable = sin(emb)
        var h = x
        for layer in layers {
            h = layer(h, cos: cosTable, sin: sinTable)
        }
        return norm(h)
    }
}
