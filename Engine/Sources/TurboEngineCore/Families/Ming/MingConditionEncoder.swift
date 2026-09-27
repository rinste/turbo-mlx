import Foundation
import MLX
import MLXNN

/// The `mlp/` folder, after mflux's `MingHeads`: the learned query tokens plus the projections
/// around the connector and the direct-VLM head that turns three encoder hidden states into
/// DiT-width caption tokens. Never quantized.
public final class MingHeads: Module {
    @ParameterInfo(key: "query_tokens") var queryTokens: MLXArray
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    /// An RMSNorm then a linear.
    @ModuleInfo(key: "proj_directvlm") var projDirectVLM: [UnaryLayer]

    public init(config: MingConfig) {
        let heads = config.heads
        _queryTokens.wrappedValue = MLXArray.zeros([heads.queryCount, config.encoder.hiddenSize])
        _projIn.wrappedValue = Linear(config.encoder.hiddenSize, config.connector.hiddenSize, bias: true)
        _projOut.wrappedValue = Linear(config.connector.hiddenSize, heads.capFeatDim, bias: true)
        let directDim = config.encoder.hiddenSize * heads.directVLMLayers.count
        _projDirectVLM.wrappedValue = [RMSNorm(dimensions: directDim, eps: 1e-5), Linear(directDim, heads.ditDim, bias: true)]
        super.init()
    }
}

/// The text side of Ming-Image as one operation: the prompt through the encoder with the query
/// tokens appended, the query states through the connector and the caption projection, the
/// prompt's hidden states through the direct-VLM head (mflux's `MingConditionEncoder`).
public struct MingConditionEncoder {
    public let config: MingConfig
    public let encoder: LingMoeEncoder
    public let connector: MingConnector
    public let heads: MingHeads

    public init(config: MingConfig, encoder: LingMoeEncoder, connector: MingConnector, heads: MingHeads) {
        self.config = config
        self.encoder = encoder
        self.connector = connector
        self.heads = heads
    }

    /// The token ids with the query block appended, the 3D `video_rope` positions
    /// `get_t_scale_rope_index` assigns (text counts up on all three axes, the 1 × 1 × n query grid
    /// shares t = h = n_text with w centred on it, the closing token resumes at t + 1), and the
    /// mask of the query tokens.
    func buildInputs(promptIds: [Int]) -> (inputIds: MLXArray, positionIds: MLXArray, imageMask: MLXArray) {
        let count = config.heads.queryCount
        let ids = promptIds + [config.imageStartId] + Array(repeating: config.imagePatchId, count: count) + [config.imageEndId]
        let textCount = promptIds.count + 1
        var t: [Int32] = []
        var h: [Int32] = []
        var w: [Int32] = []
        for index in 0 ..< textCount {
            t.append(Int32(index)); h.append(Int32(index)); w.append(Int32(index))
        }
        for index in 0 ..< count {
            t.append(Int32(textCount)); h.append(Int32(textCount)); w.append(Int32(index - (count - 1) / 2 + textCount))
        }
        t.append(Int32(textCount + 1)); h.append(Int32(textCount + 1)); w.append(Int32(textCount + 1))
        let positionIds = MLXArray(t + h + w, [3, ids.count])
        let inputIds = MLXArray(ids.map { Int32($0) })
        let imageMask = (inputIds .== MLXArray(Int32(config.imagePatchId))).expandedDimensions(axis: 0)
        return (inputIds, positionIds, imageMask)
    }

    /// Returns (capFeats [queries, capFeatDim], capFeats2 [prompt tokens, ditDim]).
    public func encode(promptIds: [Int]) -> (MLXArray, MLXArray) {
        let (inputIds, positionIds, imageMask) = buildInputs(promptIds: promptIds)
        var embeds = encoder.wordEmbeddings(inputIds)
        let queryStart = promptIds.count + 1
        let count = config.heads.queryCount
        embeds = concatenated([
            embeds[0 ..< queryStart],
            heads.queryTokens.asType(embeds.dtype),
            embeds[(queryStart + count)...],
        ], axis: 0).expandedDimensions(axis: 0)
        let layers = config.heads.directVLMLayers
        let hidden = encoder(inputsEmbeds: embeds, positionIds: positionIds, imageMask: imageMask, outputLayers: Set(layers + [config.encoder.numLayers]))

        let promptCount = promptIds.count
        let directVLM = concatenated(layers.map { hidden[$0]![0, 0 ..< promptCount] }, axis: -1)
        let capFeats2 = projDirectVLM(directVLM)

        let queryStates = hidden[config.encoder.numLayers]![0..., queryStart ..< (queryStart + count)]
        let capFeats = heads.projOut(connector(heads.projIn(queryStates)))[0]
        return (capFeats, capFeats2)
    }

    private func projDirectVLM(_ x: MLXArray) -> MLXArray {
        heads.projDirectVLM[1](heads.projDirectVLM[0](x))
    }
}
