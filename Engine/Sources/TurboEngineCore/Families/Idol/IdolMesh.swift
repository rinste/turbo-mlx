import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The Gaussians as a plain triangle GLB any viewer draws solid, as mlx-serve's `splat.zig` writes it (`writeDiscGlb`):
/// one primitive (mode 4, POSITION and TEXCOORD_0, uint32 indices), one opaque quad per Gaussian of at least 5%
/// opacity, in the plane of its two longest axes, reaching 1.5 standard deviations along each (at most 0.012), its
/// colour one texel of a PNG atlas the four corners all point at (sampler NEAREST, clamped), the material unlit
/// (`KHR_materials_unlit`), double-sided, metallic 0, roughness 1. In glTF's y-up, turned about +Y so that the camera
/// Idol predicted for the picture looks from +Z (the picture's view is the object's front), centred on the origin.
public enum IdolMesh {
    static let minimumOpacity: Float = 0.05
    static let reach: Float = 1.5
    static let maximumReach: Float = 0.012
    /// Degree-0 spherical harmonics' constant: colour = 0.5 + SH0 · coefficient.
    static let sh0: Float = 0.28209479177387814

    /// The turn about +Y bringing the camera's direction [dx, dy, dz] (the model's z-up frame, from the object to the
    /// camera) in front, as (cos, sin) of the angle.
    static func frontTurn(camera: [Float]) -> (Float, Float) {
        // In glTF's frame the direction is (dx, dz, −dy); its horizontal part (dx, −dy) goes to +Z.
        let (x, z) = (camera[0], -camera[1])
        let length = (x * x + z * z).squareRoot()
        guard length > 1e-6 else { return (1, 0) }
        return (z / length, x / length)
    }

    public struct Written {
        public let discs: Int
        public let bytes: Int
    }

    /// Writes the GLB at `url` whole or not at all (beside it, then renamed over it).
    @discardableResult
    public static func write(_ g: IdolGaussians, camera: [Float]?, to url: URL) throws -> Written {
        let kept = (0 ..< g.count).filter { g.opacity[$0] >= minimumOpacity }
        guard !kept.isEmpty else { throw IdolPicture.PictureError.empty }
        let n = kept.count
        let side = Int(Double(n - 1).squareRoot()) + 1
        let (turnCos, turnSin) = camera.map { frontTurn(camera: $0) } ?? (1, 0)
        var positions = [Float](repeating: 0, count: n * 12)
        var uv = [Float](repeating: 0, count: n * 8)
        var indices = [UInt32](repeating: 0, count: n * 6)
        var atlas = [UInt8](repeating: 0, count: side * side * 3)
        let corners: [(Float, Float)] = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for (k, i) in kept.enumerated() {
            let (w, x, y, z) = (g.rotation[4 * i], g.rotation[4 * i + 1], g.rotation[4 * i + 2], g.rotation[4 * i + 3])
            // The rotation's columns: the Gaussian's axes.
            let axes: [[Float]] = [
                [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
                [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
                [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
            ]
            let scale = [g.scale[3 * i], g.scale[3 * i + 1], g.scale[3 * i + 2]]
            let order = [0, 1, 2].sorted { scale[$0] > scale[$1] || (scale[$0] == scale[$1] && $0 < $1) }
            let ra = Swift.min(reach * scale[order[0]], maximumReach)
            let rb = Swift.min(reach * scale[order[1]], maximumReach)
            let (a, b) = (axes[order[0]], axes[order[1]])
            for (c, corner) in corners.enumerated() {
                var p = [Float](repeating: 0, count: 3)
                for d in 0 ..< 3 { p[d] = g.xyz[3 * i + d] + corner.0 * ra * a[d] + corner.1 * rb * b[d] }
                // z-up to glTF's y-up, (x, y, z) → (x, z, −y), then the turn about +Y.
                let (gx, gy, gz) = (p[0], p[2], -p[1])
                let o = (k * 4 + c) * 3
                positions[o] = gx * turnCos - gz * turnSin
                positions[o + 1] = gy
                positions[o + 2] = gx * turnSin + gz * turnCos
                uv[(k * 4 + c) * 2] = (Float(k % side) + 0.5) / Float(side)
                uv[(k * 4 + c) * 2 + 1] = (Float(k / side) + 0.5) / Float(side)
            }
            let base = UInt32(k * 4)
            indices.replaceSubrange((k * 6) ..< (k * 6 + 6), with: [base, base + 1, base + 2, base + 1, base + 3, base + 2])
            for d in 0 ..< 3 {
                let colour = Swift.min(Swift.max(0.5 + sh0 * g.colour[3 * i + d], 0), 1)
                atlas[k * 3 + d] = UInt8((colour * 255).rounded())
            }
        }
        // Centred on the origin: the middle of the corners' bounding box.
        var low = [Float](repeating: .infinity, count: 3), high = [Float](repeating: -.infinity, count: 3)
        for v in 0 ..< n * 4 {
            for d in 0 ..< 3 {
                low[d] = Swift.min(low[d], positions[3 * v + d])
                high[d] = Swift.max(high[d], positions[3 * v + d])
            }
        }
        let centre = (0 ..< 3).map { (low[$0] + high[$0]) / 2 }
        for v in 0 ..< n * 4 {
            for d in 0 ..< 3 { positions[3 * v + d] -= centre[d] }
        }
        let png = try pngData(rgb: atlas, side: side)
        let glb = container(positions: positions, uv: uv, indices: indices, png: png, vertices: n * 4,
                            low: (0 ..< 3).map { low[$0] - centre[$0] }, high: (0 ..< 3).map { high[$0] - centre[$0] })
        let staging = url.deletingLastPathComponent().appending(path: ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        do {
            try glb.write(to: staging)
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.replaceItemAt(url, withItemAt: staging)
            } else {
                try FileManager.default.moveItem(at: staging, to: url)
            }
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }
        return Written(discs: n, bytes: glb.count)
    }

    static func pngData(rgb: [UInt8], side: Int) throws -> Data {
        let data = NSMutableData()
        guard let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: side * 3,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        return data as Data
    }

    static func container(positions: [Float], uv: [Float], indices: [UInt32], png: Data, vertices: Int, low: [Float], high: [Float]) -> Data {
        var binary = Data()
        var views: [[String: Int]] = []
        func append(_ blob: Data) {
            binary.append(Data(repeating: 0, count: (4 - binary.count % 4) % 4))
            views.append(["buffer": 0, "byteOffset": binary.count, "byteLength": blob.count])
            binary.append(blob)
        }
        append(positions.withUnsafeBufferPointer { Data(buffer: $0) })
        append(uv.withUnsafeBufferPointer { Data(buffer: $0) })
        append(indices.withUnsafeBufferPointer { Data(buffer: $0) })
        append(png)
        binary.append(Data(repeating: 0, count: (4 - binary.count % 4) % 4))
        let gltf: [String: Any] = [
            "asset": ["version": "2.0", "generator": "turbo-engine \(Engine.version) idol"],
            "extensionsUsed": ["KHR_materials_unlit"],
            "scene": 0, "scenes": [["nodes": [0]]], "nodes": [["mesh": 0]],
            "meshes": [["primitives": [["mode": 4, "attributes": ["POSITION": 0, "TEXCOORD_0": 1], "indices": 2, "material": 0]]]],
            "materials": [["pbrMetallicRoughness": ["baseColorTexture": ["index": 0], "metallicFactor": 0, "roughnessFactor": 1],
                           "doubleSided": true, "extensions": ["KHR_materials_unlit": [String: Any]()]]],
            "textures": [["source": 0, "sampler": 0]],
            "images": [["bufferView": 3, "mimeType": "image/png"]],
            "samplers": [["magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071]],
            "buffers": [["byteLength": binary.count]],
            "bufferViews": views,
            "accessors": [
                ["bufferView": 0, "componentType": 5126, "count": vertices, "type": "VEC3", "min": low, "max": high],
                ["bufferView": 1, "componentType": 5126, "count": vertices, "type": "VEC2"],
                ["bufferView": 2, "componentType": 5125, "count": indices.count, "type": "SCALAR"],
            ],
        ]
        var text = (try? JSONSerialization.data(withJSONObject: gltf, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
        text.append(Data(repeating: 0x20, count: (4 - text.count % 4) % 4))
        var out = Data()
        func u32(_ value: Int) { withUnsafeBytes(of: UInt32(value).littleEndian) { out.append(contentsOf: $0) } }
        u32(0x4654_6C67); u32(2); u32(12 + 8 + text.count + 8 + binary.count)
        u32(text.count); u32(0x4E4F_534A); out.append(text)
        u32(binary.count); u32(0x004E_4942); out.append(binary)
        return out
    }
}
