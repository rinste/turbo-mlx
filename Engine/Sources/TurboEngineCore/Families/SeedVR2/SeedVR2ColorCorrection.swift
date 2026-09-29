import Foundation

/// `SeedVR2Util.apply_color_correction`, in float32 as mflux runs it with numpy: the upscaled
/// image's fine detail (a five-level à trous wavelet) over the picture's coarse colors, then its
/// a and b channels (Lab) histogram-matched to the picture's and its lightness 80% its own, 20%
/// matched. Images are interleaved RGB, [H, W, 3] in [-1, 1].
public enum SeedVR2ColorCorrection {
    static let luminanceWeight: Float = 0.8

    public static func apply(content: [Float], style: [Float], width: Int, height: Int) -> [Float] {
        let n = width * height
        let contentPlanes = planar(content, count: n)
        let stylePlanes = planar(style, count: n)

        // Wavelet reconstruction: the content's high frequencies on the style's low ones.
        var rgbContent: [[Float]] = []
        var rgbStyle: [[Float]] = []
        for channel in 0 ..< 3 {
            let high = decomposition(contentPlanes[channel], width: width, height: height).high
            let low = decomposition(stylePlanes[channel], width: width, height: height).low
            var merged = [Float](repeating: 0, count: n)
            for i in 0 ..< n { merged[i] = min(max(high[i] + low[i], -1), 1) }
            rgbContent.append(merged.map { min(max(($0 + 1) * 0.5, 0), 1) })
            rgbStyle.append(stylePlanes[channel].map { min(max(($0 + 1) * 0.5, 0), 1) })
        }

        let contentLab = rgbToLab(rgbContent, count: n)
        let styleLab = rgbToLab(rgbStyle, count: n)
        let matchedA = histogramMatch(contentLab[1], reference: styleLab[1])
        let matchedB = histogramMatch(contentLab[2], reference: styleLab[2])
        let matchedL = histogramMatch(contentLab[0], reference: styleLab[0])
        let weight = luminanceWeight
        let rest = Float(1.0 - Double(luminanceWeight))
        var lightness = [Float](repeating: 0, count: n)
        for i in 0 ..< n { lightness[i] = weight * contentLab[0][i] + rest * matchedL[i] }

        let rgb = labToRGB([lightness, matchedA, matchedB], count: n)
        var out = [Float](repeating: 0, count: 3 * n)
        for i in 0 ..< n {
            for channel in 0 ..< 3 {
                out[3 * i + channel] = min(max(rgb[channel][i], 0), 1) * 2 - 1
            }
        }
        return out
    }

    static func planar(_ interleaved: [Float], count n: Int) -> [[Float]] {
        (0 ..< 3).map { channel in (0 ..< n).map { interleaved[3 * $0 + channel] } }
    }

    // MARK: Wavelets

    /// `_wavelet_blur`: a 3 × 3 binomial kernel with taps `radius` apart, edges repeated.
    static func blur(_ image: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        var radius = max(radius, 1)
        radius = min(radius, max(1, min(height, width) / 8))
        let kernel: [[Float]] = [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
        var out = [Float](repeating: 0, count: image.count)
        for (ky, dy) in [-1, 0, 1].enumerated() {
            for (kx, dx) in [-1, 0, 1].enumerated() {
                let k = kernel[ky][kx]
                for y in 0 ..< height {
                    let sy = min(max(y + dy * radius, 0), height - 1)
                    let row = sy * width
                    let outRow = y * width
                    for x in 0 ..< width {
                        let sx = min(max(x + dx * radius, 0), width - 1)
                        out[outRow + x] += k * image[row + sx]
                    }
                }
            }
        }
        return out
    }

    /// `_wavelet_decomposition` with five levels: the sum of the details and what is left.
    static func decomposition(_ image: [Float], width: Int, height: Int, levels: Int = 5) -> (high: [Float], low: [Float]) {
        var high = [Float](repeating: 0, count: image.count)
        var current = image
        for level in 0 ..< levels {
            let low = blur(current, width: width, height: height, radius: 1 << level)
            for i in 0 ..< image.count { high[i] += current[i] - low[i] }
            current = low
        }
        return (high, current)
    }

    // MARK: Lab

    static let toXYZ: [[Float]] = [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
    static let fromXYZ: [[Float]] = [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
    static let epsilon = Float(6.0 / 29.0)
    static let epsilonCubed = Float(pow(6.0 / 29.0, 3))
    static let kappa = Float(pow(29.0 / 3.0, 3))

    static func rgbToLab(_ rgb: [[Float]], count n: Int) -> [[Float]] {
        var lightness = [Float](repeating: 0, count: n)
        var a = [Float](repeating: 0, count: n)
        var b = [Float](repeating: 0, count: n)
        let m = toXYZ
        func linear(_ x: Float) -> Float { x > 0.04045 ? pow((x + 0.055) / 1.055, 2.4) : x / 12.92 }
        func f(_ t: Float) -> Float { t > epsilonCubed ? cbrt(t) : (kappa * t + 16) / 116 }
        for i in 0 ..< n {
            let (r, g, bl) = (linear(rgb[0][i]), linear(rgb[1][i]), linear(rgb[2][i]))
            let x = (r * m[0][0] + g * m[0][1] + bl * m[0][2]) / 0.95047
            let y = r * m[1][0] + g * m[1][1] + bl * m[1][2]
            let z = (r * m[2][0] + g * m[2][1] + bl * m[2][2]) / 1.08883
            let (fx, fy, fz) = (f(x), f(y), f(z))
            lightness[i] = 116 * fy - 16
            a[i] = 500 * (fx - fy)
            b[i] = 200 * (fy - fz)
        }
        return [lightness, a, b]
    }

    static func labToRGB(_ lab: [[Float]], count n: Int) -> [[Float]] {
        var rgb = [[Float]](repeating: [Float](repeating: 0, count: n), count: 3)
        let m = fromXYZ
        let exponent = Float(1.0 / 2.4)
        func inverse(_ f: Float) -> Float { f > epsilon ? pow(f, 3) : (116 * f - 16) / kappa }
        func gamma(_ v: Float) -> Float { v > 0.0031308 ? 1.055 * pow(max(v, 0), exponent) - 0.055 : 12.92 * v }
        for i in 0 ..< n {
            let fy = (lab[0][i] + 16) / 116
            let fx = lab[1][i] / 500 + fy
            let fz = fy - lab[2][i] / 200
            let (x, y, z) = (inverse(fx) * 0.95047, inverse(fy), inverse(fz) * 1.08883)
            for row in 0 ..< 3 {
                rgb[row][i] = gamma(x * m[row][0] + y * m[row][1] + z * m[row][2])
            }
        }
        return rgb
    }

    /// `_hist_match`: each value replaced by the reference's value of the same rank (ties ranked
    /// by position, as a stable argsort ranks them).
    static func histogramMatch(_ source: [Float], reference: [Float]) -> [Float] {
        let order = source.indices.sorted { a, b in
            source[a] != source[b] ? source[a] < source[b] : a < b
        }
        let sortedReference = reference.sorted()
        var out = [Float](repeating: 0, count: source.count)
        for (rank, index) in order.enumerated() { out[index] = sortedReference[rank] }
        return out
    }
}
