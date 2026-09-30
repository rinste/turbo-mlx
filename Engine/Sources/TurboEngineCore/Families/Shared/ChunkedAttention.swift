import Foundation
import MLX

/// Single-head attention over every pixel of a frame, as the VAE mid blocks run it, with its
/// temporaries bounded. MLX's fused attention kernel takes head dimensions up to 256 and a mid
/// block's is its channel count (384 in the Qwen VAE, 512 in FLUX's), so there
/// `scaledDotProductAttention` falls back to the scores in full: [pixels, pixels] in float32,
/// 1 GB at 1024 × 1024 (128² latents) and their softmax as much again, 4 GB each at 2048. A chunk
/// of query rows at a time is the same math (each row's softmax is its own) within `budget`
/// bytes, and the result the same to rounding.
enum ChunkedAttention {
    /// The most bytes one chunk's scores take (their softmax takes as much again).
    static let budget = 256 << 20

    /// `queries`, `keys` and `values` [B, N, C], one head. The scores are computed in the inputs'
    /// dtype and scaled by `scale` as a float32 array, so a bf16 input goes on in float32 from
    /// here, as mflux's explicit attention does (Ming-Image's decode), and a float32 one stays so.
    static func attend(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float) -> MLXArray {
        let count = queries.shape[1]
        let keysT = keys.transposed(0, 2, 1)
        let scale32 = MLXArray(scale)
        func attended(_ rows: MLXArray) -> MLXArray {
            matmul(softmax(matmul(rows, keysT) * scale32, axis: -1), values)
        }
        let rowsPerChunk = max(1, budget / (count * 4))
        guard count > rowsPerChunk else { return attended(queries) }
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: count, by: rowsPerChunk) {
            let part = attended(queries[0..., start ..< min(start + rowsPerChunk, count), 0...])
            // Materialized here, so one chunk's scores are gone before the next chunk's are made.
            eval(part)
            parts.append(part)
        }
        return concatenated(parts, axis: 1)
    }
}
