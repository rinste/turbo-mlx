// Draws the app icon: a white line sparkle on a dark field, full-bleed (macOS rounds the corners).
// Every size is drawn from the same vector, with a heavier line and one star less where the icon
// gets small, and written into the asset catalog.
//
//   swiftc -O scripts/make-icon.swift -o /tmp/make-icon && /tmp/make-icon
//   /tmp/make-icon preview.png 1024      (one size, anywhere, to look at)

import AppKit
import CoreGraphics

let sizes = [16, 32, 64, 128, 256, 512, 1024]
let catalog = "TurboMLX/Resources/Assets.xcassets/AppIcon.appiconset"

/// A four-pointed star whose sides curve in towards the middle: tips at `radius` (up, down) and
/// `radius * width` (left, right). Each side leaves a tip almost along its axis (`near`) and bends
/// round towards the centre (`far`): the smaller they are, the thinner the waist.
func sparkle(center: CGPoint, radius: CGFloat, width: CGFloat, near: CGFloat = 0.04, far: CGFloat = 0.30) -> CGPath {
    let (rx, ry) = (radius * width, radius)
    func point(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: center.x + x * rx, y: center.y + y * ry) }
    let path = CGMutablePath()
    path.move(to: point(0, 1))
    // Clockwise from the top: each quadrant from one tip to the next.
    for (from, to) in [((0.0, 1.0), (1.0, 0.0)), ((1.0, 0.0), (0.0, -1.0)), ((0.0, -1.0), (-1.0, 0.0)), ((-1.0, 0.0), (0.0, 1.0))] {
        let (fx, fy) = from, (tx, ty) = to
        // Leaving `from` along its own axis, arriving at `to` along its axis.
        let c1 = point(fx == 0 ? near * tx : far * fx, fy == 0 ? near * ty : far * fy)
        let c2 = point(tx == 0 ? near * fx : far * tx, ty == 0 ? near * fy : far * ty)
        path.addCurve(to: point(tx, ty), control1: c1, control2: c2)
    }
    path.closeSubpath()
    return path
}

/// The star's outline drawn inside it, so its points stay sharp, with a faint halo.
func drawOutline(_ path: CGPath, width: CGFloat, glow: CGFloat, in context: CGContext) {
    context.saveGState()
    context.setShadow(offset: .zero, blur: glow, color: CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.28))
    context.beginTransparencyLayer(auxiliaryInfo: nil)
    context.addPath(path)
    context.clip()
    context.setLineWidth(width * 2)
    context.setLineJoin(.round)
    context.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    context.addPath(path)
    context.strokePath()
    context.endTransparencyLayer()
    context.restoreGState()
}

func render(size: Int) -> CGImage {
    let side = CGFloat(size)
    let space = CGColorSpace(name: CGColorSpace.displayP3)!
    let context = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setShouldAntialias(true)
    context.interpolationQuality = .high

    // A dark field, a little lighter behind the star.
    let field = CGGradient(colorsSpace: space, colors: [
        CGColor(srgbRed: 0.105, green: 0.106, blue: 0.125, alpha: 1),
        CGColor(srgbRed: 0.035, green: 0.035, blue: 0.045, alpha: 1),
    ] as CFArray, locations: [0, 1])!
    context.drawLinearGradient(field, start: CGPoint(x: 0, y: side), end: CGPoint(x: side, y: 0), options: [])
    let glow = CGGradient(colorsSpace: space, colors: [
        CGColor(srgbRed: 0.42, green: 0.44, blue: 0.60, alpha: 0.22),
        CGColor(srgbRed: 0.42, green: 0.44, blue: 0.60, alpha: 0),
    ] as CFArray, locations: [0, 1])!
    let middle = CGPoint(x: side * 0.47, y: side * 0.47)
    context.drawRadialGradient(glow, startCenter: middle, startRadius: 0, endCenter: middle, endRadius: side * 0.52, options: [])

    // The line: heavier, relative to the icon, as it gets small, so it still reads at 16 px.
    let line = max(side * 0.016, min(1.3, side * 0.09))
    let main = sparkle(center: middle, radius: side * 0.33, width: 0.76)
    drawOutline(main, width: line, glow: side * 0.02, in: context)

    // A small solid one up to the right, where there is room for it.
    if size >= 32 {
        let small = sparkle(center: CGPoint(x: side * 0.745, y: side * 0.755), radius: side * 0.075, width: 0.76)
        context.saveGState()
        context.setShadow(offset: .zero, blur: side * 0.015, color: CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.35))
        context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
        context.addPath(small)
        context.fillPath()
        context.restoreGState()
    }
    return context.makeImage()!
}

func write(_ image: CGImage, to path: String) {
    let bitmap = NSBitmapImageRep(cgImage: image)
    guard let data = bitmap.representation(using: .png, properties: [:]) else { fatalError("no PNG for \(path)") }
    try! data.write(to: URL(fileURLWithPath: path))
}

let arguments = CommandLine.arguments.dropFirst()
if let path = arguments.first {
    write(render(size: Int(arguments.dropFirst().first ?? "1024") ?? 1024), to: path)
} else {
    for size in sizes {
        write(render(size: size), to: "\(catalog)/icon_\(size).png")
    }
    print("Wrote \(sizes.count) sizes to \(catalog)")
}
