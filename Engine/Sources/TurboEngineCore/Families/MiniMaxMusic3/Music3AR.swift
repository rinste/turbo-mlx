import Foundation
import MLX
import MLXNN

/// The hierarchical autoregressive stage (`ar.py`): frame by frame, the global model samples the semantic code with
/// guidance (its conditional and unconditional prompts side by side in a batch of two), the depth decoder the seven
/// residual codes, and the frame's codes, embedded, feed the next step. Each frame's hidden states (the global model's
/// and the depth decoder's seven) are what the flow reads. The draws follow mlx-audio's keys exactly: the seed's key
/// split once per frame, then chained from draw to draw within it, so a seed gives mlx-audio's frames as long as the
/// two compute the same logits.
public struct Music3AR {
    public let languageModel: Music3LanguageModel
    public let depth: Music3DepthDecoder
    public let config: Music3Config

    /// Top-k sampling (`sample_top_k`): NaN as -1e9, everything below the k-th logit masked, a categorical draw with
    /// `key`; the draw and the key for the next one.
    public static func sampleTopK(_ logits: MLXArray, key: MLXArray, k: Int = Music3Config.arTopK) -> (MLXArray, MLXArray) {
        let values = MLX.where(isNaN(logits), MLXArray(Float(-1e9)).asType(logits.dtype), logits).asType(.float32)
        let threshold = top(values, k: min(k, values.dim(-1)), axis: -1).min(axis: -1, keepDims: true)
        let masked = MLX.where(values .< threshold, MLXArray(Float(-1e9)), values)
        let next = MLXRandom.split(key: key).1
        return (MLXRandom.categorical(masked, axis: -1, key: key), next)
    }

    /// The prompt's last hidden states [2, hidden] and the caches holding it.
    public func prefill(ids: MLXArray) -> (MLXArray, [Music3KVCache]) {
        let caches = languageModel.makeCaches()
        return (step(languageModel.model.embedTokens(ids), caches: caches), caches)
    }

    /// The global model's last hidden states [2, hidden] after reading `embeddings` [2, L, hidden].
    public func step(_ embeddings: MLXArray, caches: [Music3KVCache]) -> MLXArray {
        languageModel.model(embeddings, caches: caches)[0..., -1]
    }

    /// The semantic code's logits [2, vocab] guided and masked to the audio codes and the end token (`ar_one_frame`).
    public func semanticLogits(_ lastHidden: MLXArray) -> MLXArray {
        var logits = languageModel.logits(lastHidden).asType(.float32)
        let ids = MLXArray(Int32(0) ..< Int32(logits.dim(-1)))
        let offset = Int32(config.audioCodeOffset)
        let allowed = logicalOr(logicalAnd(ids .>= offset, ids .< offset + Int32(config.semanticVocabSize)),
                                ids .== Int32(config.audioEndTokenID)).expandedDimensions(axis: 0)
        let blocked = MLXArray(Float(-1e9))
        logits = MLX.where(allowed, logits, blocked)
        let conditional = logits[0 ..< 1], unconditional = logits[1 ..< 2]
        var guided = unconditional + (conditional - unconditional) * Music3Config.arGuidance
        let threshold = top(conditional, k: min(Music3Config.arTopK, conditional.dim(-1)), axis: -1).min(axis: -1, keepDims: true)
        guided = MLX.where(conditional .< threshold, blocked, guided)
        return MLX.where(allowed, guided, blocked)
    }

    /// The residual codes after `semanticCode` [2] (`generate_depth_codes`): the frame's codes [2, codebooks], the depth
    /// decoder's hidden states of the conditional row [1, 7 × hidden], and the key after the last draw.
    public func depthCodes(lastHidden: MLXArray, semanticCode: MLXArray, key: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        var sequence = [depth.projection(lastHidden).expandedDimensions(axis: 1)]
        let codeEmbedding = languageModel.model.embedTokens(semanticCode + Int32(config.audioCodeOffset))
        sequence.append(depth.projection(codeEmbedding).expandedDimensions(axis: 1))
        var codes = [semanticCode]
        var hiddenParts: [MLXArray] = []
        var key = key
        for index in 1 ..< config.numCodebooks {
            let hidden = depth(concatenated(sequence, axis: 1))[0..., -1]
            hiddenParts.append(hidden[0 ..< 1])
            let logits = depth.audioHeads[index - 1](hidden)
            let conditional = logits[0 ..< 1].asType(.float32), unconditional = logits[1 ..< 2].asType(.float32)
            let guided = unconditional + (conditional - unconditional) * Music3Config.arGuidance
            let sampled: MLXArray
            (sampled, key) = Self.sampleTopK(guided, key: key)
            let code = concatenated([sampled, sampled]).asType(.int32)
            codes.append(code)
            if index < config.numCodebooks - 1 {
                let embedding = depth.audioEmbeddings(code + Int32((index - 1) * config.audioVocabSize))
                sequence.append(depth.projection(embedding).expandedDimensions(axis: 1))
            }
        }
        return (stacked(codes, axis: 1), concatenated(hiddenParts, axis: -1), key)
    }

    /// The depth decoder's hidden states of the conditional row [1, 7 × hidden] for given codes [2, codebooks]: the draws
    /// of `depthCodes` replaced by `codes` (for checks).
    public func depthHidden(lastHidden: MLXArray, codes: MLXArray) -> MLXArray {
        var sequence = [depth.projection(lastHidden).expandedDimensions(axis: 1)]
        let codes = codes.asType(.int32)
        sequence.append(depth.projection(languageModel.model.embedTokens(codes[0..., 0] + Int32(config.audioCodeOffset))).expandedDimensions(axis: 1))
        var parts: [MLXArray] = []
        for index in 1 ..< config.numCodebooks {
            parts.append(depth(concatenated(sequence, axis: 1))[0 ..< 1, -1])
            if index < config.numCodebooks - 1 {
                let embedding = depth.audioEmbeddings(codes[0..., index] + Int32((index - 1) * config.audioVocabSize))
                sequence.append(depth.projection(embedding).expandedDimensions(axis: 1))
            }
        }
        return concatenated(parts, axis: -1)
    }

    /// The embedding the global model reads for a frame (`_embed_audio_frame`): the semantic code's, plus the residual
    /// codes' summed, over √codebooks. [2, 1, hidden].
    public func feedback(_ codes: MLXArray) -> MLXArray {
        let embeddings = languageModel.model.embedTokens(codes[0..., ..<1] + Int32(config.audioCodeOffset))
        let offsets = MLXArray(Int32(0) ..< Int32(config.residualCodebooks)) * Int32(config.audioVocabSize)
        let residual = depth.audioEmbeddings(codes[0..., 1...] + offsets.expandedDimensions(axis: 0)).sum(axis: 1, keepDims: true)
        return (embeddings + residual.asType(embeddings.dtype)) * Float(pow(Double(config.numCodebooks), -0.5))
    }

    /// Up to `maxFrames` frames' hidden states [1, frames, codebooks × hidden] for the prompt `ids` [2, tokens]
    /// (`generate_frame_hiddens`). The first step only moves past `<|audio_start|>`: its frame is not kept. The model
    /// ends the song when it samples the end token. `progress` gets the frames made so far; `isCancelled` is asked
    /// after each one.
    public func generate(
        ids: MLXArray, maxFrames: Int, seed: Int, progress: (Int) -> Void = { _ in }, isCancelled: () -> Bool = { false }
    ) throws -> MLXArray {
        var key = MLXRandom.key(UInt64(truncatingIfNeeded: seed))
        var (lastHidden, caches) = prefill(ids: ids)
        var frames: [MLXArray] = []
        for frameIndex in 0 ... maxFrames {
            let subkey: MLXArray
            (key, subkey) = MLXRandom.split(key: key)
            let (sampled, next) = Self.sampleTopK(semanticLogits(lastHidden), key: subkey)
            if sampled.item(Int.self) == config.audioEndTokenID { break }
            let semantic = concatenated([sampled, sampled]).asType(.int32) - Int32(config.audioCodeOffset)
            let (codes, depthHidden, _) = depthCodes(lastHidden: lastHidden, semanticCode: semantic, key: next)
            if frameIndex > 0 {
                let frame = concatenated([lastHidden[0 ..< 1], depthHidden], axis: -1)
                // The depth decoder's draws run while the next step's graph is built.
                asyncEval(frame)
                frames.append(frame)
                progress(frames.count)
                if frames.count >= maxFrames { break }
            }
            lastHidden = step(feedback(codes), caches: caches)
            if isCancelled() { throw GenerationError.cancelled }
        }
        guard !frames.isEmpty else { throw Music3Error.noFrames }
        return stacked(frames, axis: 1)
    }
}
