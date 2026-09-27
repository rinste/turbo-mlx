import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// PNG files from decoded pixels, with the generation's parameters in a text chunk.
public enum ImageOutput {
    public enum OutputError: LocalizedError {
        case unsupportedShape([Int])
        case cannotCreateImage
        case cannotWrite(URL)

        public var errorDescription: String? {
            switch self {
            case .unsupportedShape(let shape): "Cannot make an image from an array of shape \(shape)."
            case .cannotCreateImage: "Could not create the image."
            case .cannotWrite(let url): "Could not write \(url.path)."
            }
        }
    }

    /// `pixels` is [H, W, 3] or [H, W, 4] uint8.
    public static func cgImage(from pixels: MLXArray) throws -> CGImage {
        guard pixels.ndim == 3, pixels.dtype == .uint8, [3, 4].contains(pixels.shape[2]) else {
            throw OutputError.unsupportedShape(pixels.shape)
        }
        let (height, width, channels) = (pixels.shape[0], pixels.shape[1], pixels.shape[2])
        let data = pixels.asData(noCopy: false)
        guard let provider = CGDataProvider(data: data as CFData) else { throw OutputError.cannotCreateImage }
        let alpha: CGImageAlphaInfo = channels == 4 ? .last : .none
        guard let image = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8 * channels,
            bytesPerRow: width * channels, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: alpha.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ) else { throw OutputError.cannotCreateImage }
        return image
    }

    /// Writes a PNG; `metadata` (JSON-encodable) goes into the file's description.
    public static func writePNG(_ pixels: MLXArray, to url: URL, metadata: [String: Any]) throws {
        let image = try cgImage(from: pixels)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw OutputError.cannotWrite(url)
        }
        var properties: [CFString: Any] = [:]
        if let json = try? JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys]),
           let text = String(data: json, encoding: .utf8) {
            properties[kCGImagePropertyPNGDictionary] = [kCGImagePropertyPNGDescription: text]
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw OutputError.cannotWrite(url) }
    }
}
