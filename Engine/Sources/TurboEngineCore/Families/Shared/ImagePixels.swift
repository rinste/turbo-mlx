import Accelerate
import CoreGraphics
import Foundation
import ImageIO

/// Reading and resizing the reference images the families condition on: a clip's first frame
/// (LTX-2), the pictures an image is edited from (FLUX.2 Klein).
enum ImagePixels {
    /// The first image in the file at `path`.
    static func load(_ path: String) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }

    /// The image as RGBA bytes on an sRGB canvas, transparency over white.
    static func rgbaBytes(_ image: CGImage) -> [UInt8] {
        let (width, height) = (image.width, image.height)
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        bytes.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return bytes
    }

    /// RGBA bytes scaled to `toWidth` × `toHeight` with a Lanczos-class filter.
    static func resize(_ bytes: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        var input = bytes
        var output = [UInt8](repeating: 0, count: toWidth * toHeight * 4)
        input.withUnsafeMutableBytes { inBuffer in
            output.withUnsafeMutableBytes { outBuffer in
                var source = vImage_Buffer(data: inBuffer.baseAddress, height: vImagePixelCount(height),
                                           width: vImagePixelCount(width), rowBytes: width * 4)
                var destination = vImage_Buffer(data: outBuffer.baseAddress, height: vImagePixelCount(toHeight),
                                                width: vImagePixelCount(toWidth), rowBytes: toWidth * 4)
                _ = vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        return output
    }

    /// The RGB bytes of a `width` × `height` window at (`left`, `top`) of `rgbaWidth`-wide RGBA bytes.
    static func crop(_ rgba: [UInt8], rgbaWidth: Int, left: Int, top: Int, width: Int, height: Int) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let source = ((top + y) * rgbaWidth + (left + x)) * 4
                let target = (y * width + x) * 3
                rgb[target] = rgba[source]
                rgb[target + 1] = rgba[source + 1]
                rgb[target + 2] = rgba[source + 2]
            }
        }
        return rgb
    }
}
