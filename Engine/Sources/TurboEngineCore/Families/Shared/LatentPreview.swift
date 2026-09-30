import Foundation
import MLX

/// A glimpse of the image while it is generated: the clean image the model predicts at a step,
/// decoded small (`LatentPreview`), which the app shows in place of the empty frame.
public struct Preview {
    /// [h, w, 3] or [h, w, 4] uint8.
    public let pixels: MLXArray
    /// The step just finished, of the request's.
    public let step: Int

    public init(pixels: MLXArray, step: Int) {
        self.pixels = pixels
        self.step = step
    }
}

/// How the families make a preview cheaply: the latent grid averaged down before it goes through
/// the family's own decoder, in one piece, so a preview costs a small fraction of the decode (a
/// sixteenth at 1024 × 1024) and no new code path touches the image itself.
public enum LatentPreview {
    /// About this many pixels on the preview's longer side.
    public static let longSide = 384

    /// Whether the step just finished (`step` of `total`, from 1) shows a preview: every step of a
    /// short run, every few of a long one (six or so in all), never the last, whose image follows.
    public static func shows(step: Int, of total: Int) -> Bool {
        guard step < total else { return false }
        return step % max(1, total / 6) == 0
    }

    /// The factor a latent grid `height` × `width` (each cell `scale` pixels) is pooled by so the
    /// decoded preview's longer side is at most `longSide`: a power of two.
    public static func factor(height: Int, width: Int, scale: Int) -> Int {
        var factor = 1
        while (max(height, width) / factor) * scale > longSide, factor * 2 <= min(height, width) {
            factor *= 2
        }
        return factor
    }

    /// `grid` [1, h, w, C] averaged over `factor` × `factor` cells; the rows and columns left over
    /// at the bottom and the right are dropped.
    public static func pooled(_ grid: MLXArray, factor: Int) -> MLXArray {
        guard factor > 1, grid.shape[1] >= factor, grid.shape[2] >= factor else { return grid }
        let (height, width, channels) = (grid.shape[1] / factor * factor, grid.shape[2] / factor * factor, grid.shape[3])
        let cropped = grid[0..., 0 ..< height, 0 ..< width, 0...]
        return cropped.reshaped([1, height / factor, factor, width / factor, factor, channels]).mean(axes: [2, 4])
    }

    /// The clean image a flow model predicts from `latents` at `sigma`, given the velocity it
    /// returned (`noise` = ε − x₀ in every family here): `latents − σ · noise`, in the latents' dtype.
    public static func predicted(latents: MLXArray, noise: MLXArray, sigma: Float) -> MLXArray {
        latents - noise.asType(latents.dtype) * sigma
    }
}
