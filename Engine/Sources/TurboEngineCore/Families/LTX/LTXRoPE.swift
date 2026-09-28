import Foundation
import MLX

/// LTX's rotary embedding, SPLIT layout (dgrauet's `rope.py`): log-spaced frequencies over
/// positions normalized by a maximum per axis, the angles of every axis interleaved, zero-padded at
/// the front to half the attention width, and applied as a rotation of the two halves of each head.
struct LTXRope {
    /// [B, heads, N, headDim / 2] each, float32.
    let cos: MLXArray
    let sin: MLXArray

    /// `precompute_rope_freqs`: `positions` is [B, N, axes] (float32), `maxPositions` one per axis.
    init(positions: MLXArray, innerDim: Int, heads: Int, theta: Double, maxPositions: [Int], doublePrecision: Bool) {
        let axes = positions.shape[2]
        let (batch, count) = (positions.shape[0], positions.shape[1])
        let grid = Self.frequencyGrid(theta: theta, axes: axes, innerDim: innerDim, doublePrecision: doublePrecision)
        let frequencies = grid.shape[0]

        // Fractional positions in [0, 1], scaled to [-1, 1], times each frequency; then the axes
        // interleaved per frequency: [B, N, frequencies · axes].
        let fractions = stacked(
            (0 ..< axes).map { positions[0..., 0..., $0].asType(.float32) / Float(maxPositions[$0]) },
            axis: -1
        )
        let scaled = grid * (fractions.expandedDimensions(axis: -1) * 2 - 1)
        var angles = scaled.transposed(0, 1, 3, 2).reshaped([batch, count, frequencies * axes])

        let half = innerDim / 2
        if half > angles.shape[2] {
            angles = concatenated([MLXArray.zeros([batch, count, half - angles.shape[2]]), angles], axis: -1)
        }
        let perHead = innerDim / (2 * heads)
        cos = MLX.cos(angles).reshaped([batch, count, heads, perHead]).transposed(0, 2, 1, 3)
        sin = MLX.sin(angles).reshaped([batch, count, heads, perHead]).transposed(0, 2, 1, 3)
    }

    /// `generate_freq_grid`: θ^linspace(0, 1, n) · π/2 with n = innerDim / (2 · axes). The
    /// checkpoints ask for the power in float64 (upstream computes it with numpy): the port takes
    /// MLX's float32 linspace (i / (n − 1)), raises θ to it in float64 and casts the grid back.
    static func frequencyGrid(theta: Double, axes: Int, innerDim: Int, doublePrecision: Bool) -> MLXArray {
        let count = innerDim / (2 * axes)
        if doublePrecision {
            let values = (0 ..< count).map { i -> Float in
                let t = count > 1 ? Double(Float(i) / Float(count - 1)) : 0
                return Float(pow(theta, t) * (Double.pi / 2))
            }
            return MLXArray(values)
        }
        let exponents = MLXArray.linspace(Float(0), Float(1), count: count)
        return pow(MLXArray(Float(theta)), exponents) * Float(Double.pi / 2)
    }

    /// `apply_rope_split` on [B, heads, N, headDim]: the frequencies cast to the input's type.
    func apply(_ x: MLXArray) -> MLXArray {
        let c = cos.asType(x.dtype)
        let s = sin.asType(x.dtype)
        let half = x.shape[x.ndim - 1] / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half...]
        return concatenated([x1 * c - x2 * s, x1 * s + x2 * c], axis: -1)
    }
}
