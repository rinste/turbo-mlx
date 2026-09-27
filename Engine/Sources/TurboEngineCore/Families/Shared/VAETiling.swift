import Foundation
import MLX

/// mflux's `VAETiler.decode_image_tiled`: decodes a latent grid in 512-pixel tiles that overlap
/// by 64 pixels, blending the overlaps with cosine ramps, so a large image never needs the whole
/// decoder's activations at once. What *Save memory* does on the decoders that show no seams.
public enum VAETiling {
    /// mflux's `TilingConfig` defaults for decoding: 512-pixel tiles, an overlap of 8 latent pixels.
    public static let tileSize = 512
    public static let latentOverlap = 8

    /// `latents` is [1, h, w, C] (channels last); `decode` turns such a grid into [1, H, W, Cout].
    /// `isCancelled` is consulted between tiles.
    public static func decode(
        _ latents: MLXArray, spatialScale scale: Int = 8,
        decode: (MLXArray) throws -> MLXArray, isCancelled: () -> Bool = { false }
    ) throws -> MLXArray {
        let (latentHeight, latentWidth) = (latents.shape[1], latents.shape[2])
        let latentTile = max(1, tileSize / scale)
        guard latentHeight > latentTile || latentWidth > latentTile else {
            return try decode(latents)
        }
        let overlapPixels = latentOverlap * scale
        let overlap = max(0, min(latentOverlap, latentTile - 1))
        let stride = max(1, latentTile - overlap)
        let (height, width) = (latentHeight * scale, latentWidth * scale)
        let rampHeight = cosRamp(overlapPixels)
        let rampWidth = cosRamp(overlapPixels)

        var output: MLXArray?
        var counts: MLXArray?
        for y in Swift.stride(from: 0, to: latentHeight, by: stride) {
            let yEnd = min(y + latentTile, latentHeight)
            for x in Swift.stride(from: 0, to: latentWidth, by: stride) {
                let xEnd = min(x + latentTile, latentWidth)
                // Slivers at the end no wider than the overlap are already covered.
                if (y > 0 && yEnd - y <= overlap) || (x > 0 && xEnd - x <= overlap) { continue }
                if isCancelled() { throw GenerationError.cancelled }

                var tile = try decode(latents[0..., y ..< yEnd, x ..< xEnd, 0...]).asType(.float32)
                let (yOut, xOut) = (y * scale, x * scale)
                let tileHeight = min((yEnd - y) * scale, tile.shape[1], height - yOut)
                let tileWidth = min((xEnd - x) * scale, tile.shape[2], width - xOut)
                tile = tile[0..., 0 ..< tileHeight, 0 ..< tileWidth, 0...]
                if output == nil {
                    output = MLXArray.zeros([1, height, width, tile.shape[3]], dtype: .float32)
                    counts = MLXArray.zeros([1, height, width, 1], dtype: .float32)
                }

                let ovH = max(0, min(overlapPixels, tileHeight - 1))
                let ovW = max(0, min(overlapPixels, tileWidth - 1))
                var wh = [Float](repeating: 1, count: tileHeight)
                var ww = [Float](repeating: 1, count: tileWidth)
                if ovH > 0 {
                    if y > 0 { for i in 0 ..< ovH { wh[i] = rampHeight[i] } }
                    if yEnd < latentHeight { for i in 0 ..< ovH { wh[tileHeight - ovH + i] = 1 - rampHeight[i] } }
                }
                if ovW > 0 {
                    if x > 0 { for i in 0 ..< ovW { ww[i] = rampWidth[i] } }
                    if xEnd < latentWidth { for i in 0 ..< ovW { ww[tileWidth - ovW + i] = 1 - rampWidth[i] } }
                }
                let weights = MLXArray(wh, [1, tileHeight, 1, 1]) * MLXArray(ww, [1, 1, tileWidth, 1])
                let rows = yOut ..< (yOut + tileHeight)
                let columns = xOut ..< (xOut + tileWidth)
                output![0..., rows, columns, 0...] = output![0..., rows, columns, 0...] + tile * weights
                counts![0..., rows, columns, 0...] = counts![0..., rows, columns, 0...] + weights
                eval(output!, counts!)
            }
        }
        guard let output, let counts else { throw GenerationError.sizeTooSmall }
        return output / maximum(counts, MLXArray(Float(1e-6)))
    }

    /// `0.5 − 0.5·cos(π·t)` for `n` points from 0 to 1, as numpy's linspace spaces them.
    static func cosRamp(_ n: Int) -> [Float] {
        guard n > 0 else { return [] }
        return (0 ..< n).map { i in
            let t = n > 1 ? Float(i) / Float(n - 1) : 0
            return 0.5 - 0.5 * Float(cos(Double(t) * Double.pi))
        }
    }
}
