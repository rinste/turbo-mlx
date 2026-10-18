import CoreGraphics
import Foundation
import ImageIO
import MLX

/// The pictures a FLUX 3 image is edited from, prepared as mflux's `Flux3ReferenceImages` (which
/// mirrors BFL's `flux3.pipeline`) prepares them. One picture keeps its whole frame: scaled to fit
/// 4 MP and 4096 pixels a side, then padded by reflection to multiples of 16, the image made on that
/// canvas and cropped back to the picture. Several pictures are each scaled to fit 1 MP and cut to
/// multiples of 16 from the middle.
public enum Flux3References {
    static let ratio = 16
    static let matchMaxPixels = 4 * 1024 * 1024
    static let minSide = 64
    static let maxSide = 4096
    public static let maxReferences = 10
    static let multiReferenceMaxPixels = 1024 * 1024

    public struct Picture: Equatable {
        public let rgb: [UInt8]
        public let width: Int
        public let height: Int

        public init(rgb: [UInt8], width: Int, height: Int) {
            self.rgb = rgb
            self.width = width
            self.height = height
        }
    }

    /// The part of the canvas a single padded picture covers: left, top, right, bottom.
    public typealias Box = (left: Int, top: Int, right: Int, bottom: Int)

    public enum ReferenceError: LocalizedError {
        case tooMany(Int)

        public var errorDescription: String? {
            switch self {
            case .tooMany(let count): "FLUX 3 reads at most \(Flux3References.maxReferences) reference images, not \(count)."
            }
        }
    }

    /// The file at `path` as RGB bytes on an sRGB canvas (transparency over white), turned upright
    /// by its EXIF orientation as `ImageOps.exif_transpose` turns it.
    public static func load(_ path: String) throws -> Picture {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { throw GenerationError.unreadableImage(path) }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        let rgba = ImagePixels.rgbaBytes(image)
        let rgb = ImagePixels.crop(rgba, rgbaWidth: image.width, left: 0, top: 0, width: image.width, height: image.height)
        return oriented(Picture(rgb: rgb, width: image.width, height: image.height), orientation: orientation)
    }

    /// The picture as EXIF `orientation` says to show it: Pillow's transposes, pixel for pixel.
    static func oriented(_ picture: Picture, orientation: Int) -> Picture {
        let (w, h) = (picture.width, picture.height)
        let swaps = orientation >= 5 && orientation <= 8
        let (outWidth, outHeight) = swaps ? (h, w) : (w, h)
        // The source pixel of each output pixel (x, y).
        let source: (Int, Int) -> (Int, Int)
        switch orientation {
        case 2: source = { x, y in (w - 1 - x, y) }                 // FLIP_LEFT_RIGHT
        case 3: source = { x, y in (w - 1 - x, h - 1 - y) }         // ROTATE_180
        case 4: source = { x, y in (x, h - 1 - y) }                 // FLIP_TOP_BOTTOM
        case 5: source = { x, y in (y, x) }                         // TRANSPOSE
        case 6: source = { x, y in (y, h - 1 - x) }                 // ROTATE_270
        case 7: source = { x, y in (w - 1 - y, h - 1 - x) }         // TRANSVERSE
        case 8: source = { x, y in (w - 1 - y, x) }                 // ROTATE_90
        default: return picture
        }
        var rgb = [UInt8](repeating: 0, count: picture.rgb.count)
        for y in 0 ..< outHeight {
            for x in 0 ..< outWidth {
                let (sx, sy) = source(x, y)
                let from = (sy * w + sx) * 3
                let to = (y * outWidth + x) * 3
                rgb[to] = picture.rgb[from]
                rgb[to + 1] = picture.rgb[from + 1]
                rgb[to + 2] = picture.rgb[from + 2]
            }
        }
        return Picture(rgb: rgb, width: outWidth, height: outHeight)
    }

    /// `prepare`: one picture padded (with the box it covers), several each cut.
    public static func prepare(_ pictures: [Picture]) throws -> (pictures: [Picture], box: Box?) {
        guard !pictures.isEmpty else { throw GenerationError.referenceRequired }
        guard pictures.count <= maxReferences else { throw ReferenceError.tooMany(pictures.count) }
        if pictures.count > 1 { return (try pictures.map(prepareMultiReference), nil) }
        let (picture, box) = prepareReference(pictures[0])
        return ([picture], box)
    }

    /// `reference_canvas`: the size a single picture is scaled to (`fit`) and the canvas it is
    /// padded to, sides multiples of 16 within 64…4096, at most 4 MP in all.
    public static func referenceCanvas(width: Int, height: Int) -> (fit: (width: Int, height: Int), canvas: (width: Int, height: Int)) {
        var scale = min(1.0, (Double(matchMaxPixels) / Double(width * height)).squareRoot(), Double(maxSide) / Double(max(width, height)))
        while true {
            let fit = (width: max(1, Int((Double(width) * scale).rounded(.toNearestOrEven))),
                       height: max(1, Int((Double(height) * scale).rounded(.toNearestOrEven))))
            let canvas = (width: min(maxSide, max(minSide, ceil16(fit.width))), height: min(maxSide, max(minSide, ceil16(fit.height))))
            if canvas.width * canvas.height <= matchMaxPixels { return (fit, canvas) }
            scale *= 0.995
        }
    }

    /// `prepare_reference`: scaled to fit with Pillow's Lanczos, then padded around the middle by
    /// reflection (by symmetry when a side is 16 pixels or less) to the canvas, as `np.pad` pads.
    public static func prepareReference(_ picture: Picture) -> (Picture, Box?) {
        let (fit, canvas) = referenceCanvas(width: picture.width, height: picture.height)
        var picture = picture
        if fit != (picture.width, picture.height) {
            picture = Picture(rgb: PILResample.resize(picture.rgb, width: picture.width, height: picture.height,
                                                      toWidth: fit.width, toHeight: fit.height, filter: .lanczos),
                              width: fit.width, height: fit.height)
        }
        if canvas == (picture.width, picture.height) { return (picture, nil) }
        let left = (canvas.width - picture.width) / 2
        let top = (canvas.height - picture.height) / 2
        let reflect = min(picture.width, picture.height) > ratio
        let columns = (0 ..< canvas.width).map { padIndex($0 - left, count: picture.width, reflect: reflect) }
        var rgb = [UInt8](repeating: 0, count: canvas.width * canvas.height * 3)
        for y in 0 ..< canvas.height {
            let row = padIndex(y - top, count: picture.height, reflect: reflect) * picture.width
            for x in 0 ..< canvas.width {
                let from = (row + columns[x]) * 3
                let to = (y * canvas.width + x) * 3
                rgb[to] = picture.rgb[from]
                rgb[to + 1] = picture.rgb[from + 1]
                rgb[to + 2] = picture.rgb[from + 2]
            }
        }
        let box = (left: left, top: top, right: left + picture.width, bottom: top + picture.height)
        return (Picture(rgb: rgb, width: canvas.width, height: canvas.height), box)
    }

    /// The index `np.pad` reads for position `index` (relative to the first real one) of an axis of
    /// `count`: "reflect" mirrors around the edge pixels, "symmetric" repeats them.
    static func padIndex(_ index: Int, count: Int, reflect: Bool) -> Int {
        if reflect {
            guard count > 1 else { return 0 }
            let period = 2 * (count - 1)
            let i = abs(index) % period
            return i > count - 1 ? period - i : i
        }
        let period = 2 * count
        let i = ((index % period) + period) % period
        return i >= count ? period - 1 - i : i
    }

    /// `prepare_multi_reference`: above 1 MP scaled down (sides truncated) with Lanczos, then cut
    /// to multiples of 16 from the middle.
    public static func prepareMultiReference(_ picture: Picture) throws -> Picture {
        let sizes = multiReferenceSizes(width: picture.width, height: picture.height)
        var (width, height) = (picture.width, picture.height)
        var rgb = picture.rgb
        if sizes.scaled != (width, height) {
            rgb = PILResample.resize(rgb, width: width, height: height, toWidth: sizes.scaled.width, toHeight: sizes.scaled.height, filter: .lanczos)
            (width, height) = sizes.scaled
        }
        let (w, h) = sizes.cut
        guard w > 0, h > 0 else { throw GenerationError.referenceTooSmall }
        let (left, top) = ((width - w) / 2, (height - h) / 2)
        var cut = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0 ..< h {
            let from = ((top + y) * width + left) * 3
            cut.replaceSubrange((y * w * 3) ..< ((y + 1) * w * 3), with: rgb[from ..< (from + w * 3)])
        }
        return Picture(rgb: cut, width: w, height: h)
    }

    /// The size one of several pictures is scaled to (its own below 1 MP) and the size it is cut to.
    public static func multiReferenceSizes(width: Int, height: Int) -> (scaled: (width: Int, height: Int), cut: (width: Int, height: Int)) {
        var scaled = (width: width, height: height)
        if width * height > multiReferenceMaxPixels {
            let scale = (Double(multiReferenceMaxPixels) / Double(width * height)).squareRoot()
            scaled = (Int(Double(width) * scale), Int(Double(height) * scale))
        }
        return (scaled, (scaled.width / ratio * ratio, scaled.height / ratio * ratio))
    }

    /// `resolve_canvas`: a single padded picture is generated on its canvas and cropped back to it
    /// when the size asked for is the picture's own (what mflux does without a size); any other
    /// size is made as asked, without a crop.
    public static func resolveCanvas(_ pictures: [Picture], box: Box?, width: Int, height: Int) -> (width: Int, height: Int, crop: Box?) {
        if let box, width == box.right - box.left, height == box.bottom - box.top {
            return (pictures[0].width, pictures[0].height, box)
        }
        return (width, height, nil)
    }

    /// `to_array`: [1, H, W, 3] float32 in [-1, 1], `x / 127.5 − 1` as numpy computes it.
    public static func pixels(_ picture: Picture) -> MLXArray {
        let values = picture.rgb.map { Float($0) / Float(127.5) - 1 }
        return MLXArray(values, [1, picture.height, picture.width, 3])
    }

    static func ceil16(_ value: Int) -> Int { (value + ratio - 1) / ratio * ratio }
}
