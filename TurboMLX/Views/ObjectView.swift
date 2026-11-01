import AppKit
import ImageIO
import SceneKit
import SwiftUI
import UniformTypeIdentifiers

/// The 3D objects Idol makes, as the engine writes them: a GLB of one primitive (triangles, POSITION and TEXCOORD_0,
/// 32-bit indices), its colours one texel each of a PNG atlas, unlit, double-sided, in glTF's y-up, the picture's
/// view facing +Z, centred on the origin. Read straight into SceneKit, which has no glTF reader of its own.
nonisolated enum ObjectMesh {
    struct ReadError: LocalizedError {
        var errorDescription: String? { "The 3D object’s file could not be read." }
    }

    /// The object's node, and the radius of the sphere around the origin that holds it.
    static func node(contentsOf url: URL) throws -> (node: SCNNode, radius: Float) {
        let data = try Data(contentsOf: url)
        func u32(_ offset: Int) -> Int {
            Int(data.subdata(in: offset ..< offset + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
        }
        guard data.count > 28, u32(0) == 0x4654_6C67 else { throw ReadError() }
        let jsonLength = u32(12)
        guard let json = try JSONSerialization.jsonObject(with: data.subdata(in: 20 ..< 20 + jsonLength)) as? [String: Any],
              let views = json["bufferViews"] as? [[String: Any]], let accessors = json["accessors"] as? [[String: Any]],
              let primitive = ((json["meshes"] as? [[String: Any]])?.first?["primitives"] as? [[String: Any]])?.first,
              let attributes = primitive["attributes"] as? [String: Int],
              let positionIndex = attributes["POSITION"], let uvIndex = attributes["TEXCOORD_0"], let indexAccessor = primitive["indices"] as? Int
        else { throw ReadError() }
        let binary = data.subdata(in: 20 + jsonLength + 8 ..< data.count)
        func bytes(view: Int) throws -> Data {
            guard view < views.count, let length = views[view]["byteLength"] as? Int else { throw ReadError() }
            let offset = views[view]["byteOffset"] as? Int ?? 0
            guard offset + length <= binary.count else { throw ReadError() }
            return binary.subdata(in: offset ..< offset + length)
        }
        func accessor(_ index: Int) throws -> (data: Data, count: Int, type: Int) {
            guard index < accessors.count, let view = accessors[index]["bufferView"] as? Int,
                  let count = accessors[index]["count"] as? Int, let type = accessors[index]["componentType"] as? Int
            else { throw ReadError() }
            return (try bytes(view: view), count, type)
        }
        let positions = try accessor(positionIndex)
        let uv = try accessor(uvIndex)
        let indices = try accessor(indexAccessor)
        guard positions.type == 5126, uv.type == 5126, indices.type == 5125 || indices.type == 5123 else { throw ReadError() }

        let vertices = SCNGeometrySource(data: positions.data, semantic: .vertex, vectorCount: positions.count, usesFloatComponents: true,
                                         componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        let texture = SCNGeometrySource(data: uv.data, semantic: .texcoord, vectorCount: uv.count, usesFloatComponents: true,
                                        componentsPerVector: 2, bytesPerComponent: 4, dataOffset: 0, dataStride: 8)
        let element = SCNGeometryElement(data: indices.data, primitiveType: .triangles, primitiveCount: indices.count / 3,
                                         bytesPerIndex: indices.type == 5125 ? 4 : 2)
        let geometry = SCNGeometry(sources: [vertices, texture], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.isDoubleSided = true
        if let image = (json["images"] as? [[String: Any]])?.first, let view = image["bufferView"] as? Int,
           let source = CGImageSourceCreateWithData(try bytes(view: view) as CFData, nil),
           let atlas = CGImageSourceCreateImageAtIndex(source, 0, nil) {
            material.diffuse.contents = atlas
            material.diffuse.magnificationFilter = .nearest
            material.diffuse.minificationFilter = .nearest
            material.diffuse.mipFilter = .none
            material.diffuse.wrapS = .clamp
            material.diffuse.wrapT = .clamp
        }
        geometry.materials = [material]
        var radius: Float = 0
        positions.data.withUnsafeBytes { raw in
            let floats = raw.bindMemory(to: Float.self)
            for v in 0 ..< positions.count {
                let (x, y, z) = (floats[3 * v], floats[3 * v + 1], floats[3 * v + 2])
                radius = max(radius, (x * x + y * y + z * z).squareRoot())
            }
        }
        return (SCNNode(geometry: geometry), max(radius, 0.01))
    }

    /// A scene holding the object, with a camera looking from +Z, a little right and above, the object filling its view.
    static func scene(contentsOf url: URL) throws -> (scene: SCNScene, camera: SCNNode) {
        let (object, radius) = try node(contentsOf: url)
        let scene = SCNScene()
        scene.rootNode.addChildNode(object)
        let camera = SCNCamera()
        camera.fieldOfView = 30
        camera.zNear = Double(radius) * 0.05
        camera.zFar = Double(radius) * 20
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        let distance = radius / sin(Float.pi * 15 / 180) * 1.05
        let direction = simd_normalize(SIMD3<Float>(0.3, 0.25, 1))
        cameraNode.simdPosition = direction * distance
        cameraNode.simdLook(at: .zero)
        scene.rootNode.addChildNode(cameraNode)
        return (scene, cameraNode)
    }
}

/// The still the history shows for a 3D object: a PNG drawn from its GLB, the background left transparent.
enum ObjectStill {
    static func write(of glb: URL, to url: URL, size: Int = 768) throws {
        let (scene, camera) = try ObjectMesh.scene(contentsOf: glb)
        scene.background.contents = NSColor.clear
        let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
        renderer.scene = scene
        renderer.pointOfView = camera
        let image = renderer.snapshot(atTime: 0, with: CGSize(width: size, height: size), antialiasingMode: .multisampling4X)
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ObjectMesh.ReadError() }
        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else { throw ObjectMesh.ReadError() }
    }
}

/// A 3D object in the output panel: turned with the mouse or the trackpad, zoomed with the scroll wheel or a pinch.
struct ObjectStage: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false
        view.backgroundColor = .clear
        view.antialiasingMode = .multisampling4X
        load(into: view)
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {
        if context.coordinator.url != url { load(into: view) }
        context.coordinator.url = url
    }

    func makeCoordinator() -> Coordinator { Coordinator(url: url) }

    final class Coordinator {
        var url: URL
        init(url: URL) { self.url = url }
    }

    private func load(into view: SCNView) {
        guard let (scene, camera) = try? ObjectMesh.scene(contentsOf: url) else {
            view.scene = nil
            return
        }
        view.scene = scene
        view.pointOfView = camera
    }
}
