import Foundation
import MLX

/// From a decoder's output to the bytes of an image, as mflux's `ImageUtil` converts them.
public enum Pixels {
    /// [B, H, W, C] in [-1, 1] → [H, W, C] uint8 (C = 3 or 4). As mflux: halved, shifted and
    /// clipped in the decoder's dtype, then scaled and rounded in float32.
    public static func toPixels(_ decoded: MLXArray) -> MLXArray {
        let unit = clip(decoded / 2 + 0.5, min: 0, max: 1).asType(.float32)
        return (unit * 255).round().asType(.uint8)[0]
    }

    /// Composites an RGBA image onto white, as `MingImage.to_rgb` pastes it with its alpha.
    public static func flattenAlpha(_ pixels: MLXArray, background: Float = 255) -> MLXArray {
        guard pixels.ndim == 3, pixels.shape[2] == 4 else { return pixels }
        let rgb = pixels[.ellipsis, 0 ..< 3].asType(.float32)
        let alpha = pixels[.ellipsis, 3 ..< 4].asType(.float32) / 255
        return (rgb * alpha + background * (1 - alpha)).round().clip(min: 0, max: 255).asType(.uint8)
    }
}

extension MLXArray {
    func clip(min: Float, max: Float) -> MLXArray {
        MLX.clip(self, min: min, max: max)
    }
}
