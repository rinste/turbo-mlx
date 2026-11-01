import CoreGraphics
import Foundation
import ImageIO

/// The picture Idol is given, as Pillow holds it after `Image.open`: its stored 8-bit samples, not turned by its EXIF
/// orientation nor converted to another colour space, RGB or RGBA with straight alpha.
public struct IdolPicture {
    public var bytes: [UInt8]
    public var width: Int
    public var height: Int
    /// 3 or 4.
    public var channels: Int

    public init(bytes: [UInt8], width: Int, height: Int, channels: Int) {
        (self.bytes, self.width, self.height, self.channels) = (bytes, width, height, channels)
    }

    public enum PictureError: LocalizedError {
        case unreadable(String)
        case empty

        public var errorDescription: String? {
            switch self {
            case .unreadable(let path): "Couldn’t read the picture \((path as NSString).lastPathComponent)."
            case .empty: "No object was found in the picture: the cutout left nothing."
            }
        }
    }

    /// Any picture ImageIO reads. RGB(A) and grey samples are taken as stored; other layouts (16-bit, CMYK, indexed)
    /// are drawn into sRGB first.
    public static func load(_ path: String) throws -> IdolPicture {
        guard let image = ImagePixels.load(path) else { throw PictureError.unreadable(path) }
        if let picture = stored(image) { return picture }
        return drawn(image)
    }

    /// The samples as stored, for the layouts Pillow opens as `RGB`, `RGBA` or `L`.
    static func stored(_ image: CGImage) -> IdolPicture? {
        guard image.bitsPerComponent == 8, let data = image.dataProvider?.data, let base = CFDataGetBytePtr(data),
              let model = image.colorSpace?.model, model == .rgb || model == .monochrome
        else { return nil }
        let (width, height, rowBytes) = (image.width, image.height, image.bytesPerRow)
        let alpha = image.alphaInfo
        let little = image.bitmapInfo.contains(.byteOrder32Little)
        let pixelBytes = image.bitsPerPixel / 8
        // Where R, G, B and A sit in a pixel's bytes (nil: no alpha).
        let order: (r: Int, g: Int, b: Int, a: Int?)
        switch (model, pixelBytes, alpha, little) {
        case (.monochrome, 1, .none, _): order = (0, 0, 0, nil)
        case (.monochrome, 2, .last, false): order = (0, 0, 0, nil)  // Pillow's "LA" becomes RGB without its alpha
        case (.rgb, 3, .none, false): order = (0, 1, 2, nil)
        case (.rgb, 4, .noneSkipLast, false): order = (0, 1, 2, nil)
        case (.rgb, 4, .last, false): order = (0, 1, 2, 3)
        case (.rgb, 4, .noneSkipFirst, true): order = (2, 1, 0, nil)
        case (.rgb, 4, .first, true): order = (2, 1, 0, 3)
        case (.rgb, 4, .noneSkipFirst, false): order = (1, 2, 3, nil)
        case (.rgb, 4, .first, false): order = (1, 2, 3, 0)
        default: return nil
        }
        let channels = order.a == nil ? 3 : 4
        var bytes = [UInt8](repeating: 0, count: width * height * channels)
        for y in 0 ..< height {
            let row = base + y * rowBytes
            for x in 0 ..< width {
                let p = row + x * pixelBytes
                let o = (y * width + x) * channels
                bytes[o] = p[order.r]
                bytes[o + 1] = p[order.g]
                bytes[o + 2] = p[order.b]
                if let a = order.a { bytes[o + 3] = p[a] }
            }
        }
        return IdolPicture(bytes: bytes, width: width, height: height, channels: channels)
    }

    /// Any other layout, drawn into sRGB and un-premultiplied; RGB when it is wholly opaque.
    static func drawn(_ image: CGImage) -> IdolPicture {
        let (width, height) = (image.width, image.height)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var opaque = true
        for i in stride(from: 0, to: rgba.count, by: 4) {
            let a = Int(rgba[i + 3])
            if a < 255 { opaque = false }
            if a > 0 && a < 255 {
                for c in 0 ..< 3 { rgba[i + c] = UInt8(Swift.min(255, (Int(rgba[i + c]) * 255 + a / 2) / a)) }
            }
        }
        guard opaque else { return IdolPicture(bytes: rgba, width: width, height: height, channels: 4) }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0 ..< width * height {
            rgb[3 * i] = rgba[4 * i]
            rgb[3 * i + 1] = rgba[4 * i + 1]
            rgb[3 * i + 2] = rgba[4 * i + 2]
        }
        return IdolPicture(bytes: rgb, width: width, height: height, channels: 3)
    }

    /// `has_real_alpha`: RGBA with at least one pixel not wholly opaque.
    public var hasRealAlpha: Bool {
        guard channels == 4 else { return false }
        return stride(from: 3, to: bytes.count, by: 4).contains { bytes[$0] < 255 }
    }

    /// The RGB samples alone.
    public var rgb: [UInt8] {
        guard channels == 4 else { return bytes }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0 ..< width * height {
            rgb[3 * i] = bytes[4 * i]
            rgb[3 * i + 1] = bytes[4 * i + 1]
            rgb[3 * i + 2] = bytes[4 * i + 2]
        }
        return rgb
    }

    /// `Image.resize(LANCZOS)` as Pillow does it: RGB directly, RGBA premultiplied (`RGBa`), resized, then divided back.
    public func resized(width toWidth: Int, height toHeight: Int) -> IdolPicture {
        guard channels == 4 else {
            return IdolPicture(bytes: PILResample.resize(bytes, width: width, height: height, toWidth: toWidth, toHeight: toHeight, filter: .lanczos),
                               width: toWidth, height: toHeight, channels: 3)
        }
        var premultiplied = bytes
        for i in stride(from: 0, to: bytes.count, by: 4) {
            let a = Int(bytes[i + 3])
            for c in 0 ..< 3 { premultiplied[i + c] = Self.mulDiv255(Int(bytes[i + c]), a) }
        }
        var out = PILResample.resize(premultiplied, channels: 4, width: width, height: height, toWidth: toWidth, toHeight: toHeight, filter: .lanczos)
        for i in stride(from: 0, to: out.count, by: 4) {
            let a = Int(out[i + 3])
            if a == 0 || a == 255 { continue }
            for c in 0 ..< 3 { out[i + c] = UInt8(Swift.min(255, 255 * Int(out[i + c]) / a)) }
        }
        return IdolPicture(bytes: out, width: toWidth, height: toHeight, channels: 4)
    }

    /// Pillow's `MULDIV255`: a · b / 255, rounded its way.
    static func mulDiv255(_ a: Int, _ b: Int) -> UInt8 {
        let t = a * b + 128
        return UInt8(((t >> 8) + t) >> 8)
    }
}

/// idol-sdk's `preprocess_image`, in Pillow's arithmetic: the short side to 1024 (Lanczos); the alpha the picture has,
/// or BiRefNet's; the alpha eroded by one pixel (`MinFilter(3)`, edges repeated); a square of `margin` × the object's
/// bounding box around its centre (outside the picture: transparent black); that square to 1024 (Lanczos,
/// premultiplied); the result over black. Returns the RGB bytes, `size` × `size`.
public enum IdolPreparation {
    public static func prepare(_ picture: IdolPicture, size: Int = 1024, margin: Double,
                               matte: (_ rgb: [UInt8], _ width: Int, _ height: Int) throws -> [UInt8]) throws -> [UInt8] {
        let scale = Double(size) / Double(Swift.min(picture.width, picture.height))
        let width = Swift.max(1, Int((Double(picture.width) * scale).rounded(.toNearestOrEven)))
        let height = Swift.max(1, Int((Double(picture.height) * scale).rounded(.toNearestOrEven)))
        let resized = picture.resized(width: width, height: height)
        let alpha = try resized.hasRealAlpha
            ? stride(from: 3, to: resized.bytes.count, by: 4).map { resized.bytes[$0] }
            : matte(resized.rgb, width, height)
        let eroded = minFilter3(alpha, width: width, height: height)
        // The object's bounding box: the extent of the alpha left.
        var (left, top, right, bottom) = (width, height, -1, -1)
        for y in 0 ..< height {
            for x in 0 ..< width where eroded[y * width + x] != 0 {
                left = Swift.min(left, x); right = Swift.max(right, x)
                top = Swift.min(top, y); bottom = Swift.max(bottom, y)
            }
        }
        guard right >= 0 else { throw IdolPicture.PictureError.empty }
        let centreX = Double(left + right) / 2, centreY = Double(top + bottom) / 2
        let half = Double(Swift.max(right - left, bottom - top)) / 2 * margin
        // Python's int(): towards zero.
        let box = [Int(centreX - half), Int(centreY - half), Int(centreX + half), Int(centreY + half)]
        let (cropWidth, cropHeight) = (box[2] - box[0], box[3] - box[1])
        let rgb = resized.rgb
        var crop = [UInt8](repeating: 0, count: cropWidth * cropHeight * 4)
        for y in 0 ..< cropHeight {
            let sy = box[1] + y
            guard sy >= 0, sy < height else { continue }
            for x in 0 ..< cropWidth {
                let sx = box[0] + x
                guard sx >= 0, sx < width else { continue }
                let (s, o) = (sy * width + sx, (y * cropWidth + x) * 4)
                crop[o] = rgb[3 * s]; crop[o + 1] = rgb[3 * s + 1]; crop[o + 2] = rgb[3 * s + 2]
                crop[o + 3] = eroded[s]
            }
        }
        let square = IdolPicture(bytes: crop, width: cropWidth, height: cropHeight, channels: 4).resized(width: size, height: size)
        // `paste(image, mask=alpha)` onto black: DIV255(c · a).
        var out = [UInt8](repeating: 0, count: size * size * 3)
        for i in 0 ..< size * size {
            let a = Int(square.bytes[4 * i + 3])
            for c in 0 ..< 3 {
                let t = Int(square.bytes[4 * i + c]) * a + 128
                out[3 * i + c] = UInt8(((t >> 8) + t) >> 8)
            }
        }
        return out
    }

    /// `ImageFilter.MinFilter(3)`: the smallest value of each 3 × 3 neighbourhood, the edges repeated outwards.
    static func minFilter3(_ values: [UInt8], width: Int, height: Int) -> [UInt8] {
        var rows = [UInt8](repeating: 0, count: values.count)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let l = values[y * width + Swift.max(x - 1, 0)], r = values[y * width + Swift.min(x + 1, width - 1)]
                rows[y * width + x] = Swift.min(l, values[y * width + x], r)
            }
        }
        var out = rows
        for y in 0 ..< height {
            for x in 0 ..< width {
                let u = rows[Swift.max(y - 1, 0) * width + x], d = rows[Swift.min(y + 1, height - 1) * width + x]
                out[y * width + x] = Swift.min(u, rows[y * width + x], d)
            }
        }
        return out
    }
}
