import Foundation

/// Pillow's `Image.resize` for 8-bit RGB with its BICUBIC and LANCZOS filters, bit for bit
/// (`libImaging/Resample.c`): the coefficients in double precision, turned into 22-bit fixed point,
/// the horizontal pass first into 8-bit rows, then the vertical one. mflux prepares the picture an
/// edit starts from with it, so the engine does the same instead of an approximation.
enum PILResample {
    enum Filter {
        case bicubic
        case lanczos

        var support: Double { self == .bicubic ? 2 : 3 }

        func callAsFunction(_ x: Double) -> Double {
            switch self {
            case .bicubic:
                let a = -0.5
                let x = abs(x)
                if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
                return 0
            case .lanczos:
                guard -3 <= x, x < 3 else { return 0 }
                return Self.sinc(x) * Self.sinc(x / 3)
            }
        }

        private static func sinc(_ x: Double) -> Double {
            if x == 0 { return 1 }
            let angle = x * Double.pi
            return sin(angle) / angle
        }
    }

    private static let precisionBits = 22

    /// `rgb` holds `width` × `height` pixels, three bytes each, row after row; the result
    /// `toWidth` × `toHeight` the same way. The same size returns the bytes unchanged, as Pillow
    /// copies the image then.
    static func resize(_ rgb: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int, filter: Filter) -> [UInt8] {
        let horizontal = coefficients(inSize: width, outSize: toWidth, filter: filter)
        var vertical = coefficients(inSize: height, outSize: toHeight, filter: filter)
        var source = rgb
        var rows = height
        if toWidth != width {
            // Only the rows the vertical pass reads, their bounds shifted to the first of them.
            let first = vertical.bounds[0].start
            let last = vertical.bounds[toHeight - 1].start + vertical.bounds[toHeight - 1].count
            vertical.bounds = vertical.bounds.map { ($0.start - first, $0.count) }
            source = horizontalPass(source, width: width, firstRow: first, rows: last - first, outWidth: toWidth, horizontal)
            rows = last - first
        }
        if toHeight != height {
            source = verticalPass(source, width: toWidth, rows: rows, outHeight: toHeight, vertical)
        }
        return source
    }

    private struct Coefficients {
        var bounds: [(start: Int, count: Int)]
        /// `size` fixed-point weights per output position.
        var weights: [Int]
        var size: Int
    }

    /// `precompute_coeffs` then `normalize_coeffs_8bpc`.
    private static func coefficients(inSize: Int, outSize: Int, filter: Filter) -> Coefficients {
        let scale = Double(Float(inSize)) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = filter.support * filterScale
        let size = Int(ceil(support)) * 2 + 1
        var bounds: [(start: Int, count: Int)] = []
        var weights = [Int](repeating: 0, count: outSize * size)
        let one = Double(1 << precisionBits)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1 / filterScale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var k = [Double](repeating: 0, count: size)
            var total = 0.0
            for x in 0 ..< xmax {
                let w = filter((Double(x + xmin) - center + 0.5) * ss)
                k[x] = w
                total += w
            }
            if total != 0 {
                for x in 0 ..< xmax { k[x] /= total }
            }
            for x in 0 ..< size {
                weights[xx * size + x] = k[x] < 0 ? Int(-0.5 + k[x] * one) : Int(0.5 + k[x] * one)
            }
            bounds.append((xmin, xmax))
        }
        return Coefficients(bounds: bounds, weights: weights, size: size)
    }

    private static func clip8(_ value: Int) -> UInt8 {
        if value >= 1 << (precisionBits + 8) { return 255 }
        if value <= 0 { return 0 }
        return UInt8(value >> precisionBits)
    }

    private static func horizontalPass(_ rgb: [UInt8], width: Int, firstRow: Int, rows: Int, outWidth: Int, _ c: Coefficients) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: outWidth * rows * 3)
        let half = 1 << (precisionBits - 1)
        rgb.withUnsafeBufferPointer { input in
            out.withUnsafeMutableBufferPointer { output in
                for y in 0 ..< rows {
                    let row = (firstRow + y) * width * 3
                    for xx in 0 ..< outWidth {
                        let (start, count) = c.bounds[xx]
                        let k = xx * c.size
                        var (r, g, b) = (half, half, half)
                        for x in 0 ..< count {
                            let p = row + (start + x) * 3
                            let w = c.weights[k + x]
                            r += Int(input[p]) * w
                            g += Int(input[p + 1]) * w
                            b += Int(input[p + 2]) * w
                        }
                        let o = (y * outWidth + xx) * 3
                        output[o] = clip8(r)
                        output[o + 1] = clip8(g)
                        output[o + 2] = clip8(b)
                    }
                }
            }
        }
        return out
    }

    private static func verticalPass(_ rgb: [UInt8], width: Int, rows: Int, outHeight: Int, _ c: Coefficients) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * outHeight * 3)
        let half = 1 << (precisionBits - 1)
        rgb.withUnsafeBufferPointer { input in
            out.withUnsafeMutableBufferPointer { output in
                for yy in 0 ..< outHeight {
                    let (start, count) = c.bounds[yy]
                    let k = yy * c.size
                    for x in 0 ..< width {
                        var (r, g, b) = (half, half, half)
                        for y in 0 ..< count {
                            let p = ((start + y) * width + x) * 3
                            let w = c.weights[k + y]
                            r += Int(input[p]) * w
                            g += Int(input[p + 1]) * w
                            b += Int(input[p + 2]) * w
                        }
                        let o = (yy * width + x) * 3
                        output[o] = clip8(r)
                        output[o + 1] = clip8(g)
                        output[o + 2] = clip8(b)
                    }
                }
            }
        }
        return out
    }
}
