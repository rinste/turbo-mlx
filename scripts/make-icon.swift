// Draws the app icon: a halftone of white dots on black, largest in the middle, full-bleed
// (macOS rounds the corners). Every size is drawn from the same description, with fewer dots
// where the icon gets small, and written into the asset catalog.
//
//   swiftc -O scripts/make-icon.swift -o /tmp/make-icon && /tmp/make-icon
//   /tmp/make-icon preview.png 1024 [variant]     (one size, anywhere, to look at)

import AppKit
import CoreGraphics
import Foundation

let sizes = [16, 32, 64, 128, 256, 512, 1024]
let catalog = "TurboMLX/Resources/Assets.xcassets/AppIcon.appiconset"

/// How big each dot is, 0...1, at a grid position `(x, y)` in -1...1 from the middle.
typealias Tone = (_ x: Double, _ y: Double) -> Double

let variants: [String: Tone] = [
    // Round and centred: a glow in dots, full in the middle and falling off towards the edge.
    "center": { x, y in exp(-pow((x * x + y * y) / 0.5, 1.4)) },
    // A sphere lit from the upper left: the dots swell towards the light, and fall off at its rim.
    "sphere": { x, y in
        let r2 = x * x + y * y
        guard r2 < 0.86 else { return 0 }
        let z = (1 - r2 / 0.86).squareRoot()
        let (lx, ly, lz) = (-0.45, 0.5, 0.74)
        let lit = max(0, (x / 0.93) * lx + (y / 0.93) * ly + z * lz)
        return 0.25 + 0.75 * pow(lit, 1.4)
    },
    // Off-centre, as in the reference: a bright patch up to the right.
    "offset": { x, y in exp(-((x - 0.2) * (x - 0.2) + (y - 0.22) * (y - 0.22)) / 0.3) },
]

func render(size: Int, variant: String) -> CGImage {
    let side = CGFloat(size)
    let space = CGColorSpace(name: CGColorSpace.displayP3)!
    let context = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setShouldAntialias(true)

    // Near black, a touch lighter in the middle.
    context.setFillColor(CGColor(srgbRed: 0.025, green: 0.025, blue: 0.03, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: side, height: side))
    let lift = CGGradient(colorsSpace: space, colors: [
        CGColor(srgbRed: 0.10, green: 0.10, blue: 0.12, alpha: 1),
        CGColor(srgbRed: 0.10, green: 0.10, blue: 0.12, alpha: 0),
    ] as CFArray, locations: [0, 1])!
    let middle = CGPoint(x: side / 2, y: side / 2)
    context.drawRadialGradient(lift, startCenter: middle, startRadius: 0, endCenter: middle, endRadius: side * 0.55, options: [])

    // Fewer, larger dots as the icon shrinks, so that they stay dots.
    let count = size >= 128 ? 9 : size >= 64 ? 7 : size >= 32 ? 5 : 4
    let span = side * 0.62
    let step = span / CGFloat(count - 1)
    let origin = (side - span) / 2
    let smallest = max(step * 0.09, 0.45)
    let largest = step * 0.46
    let tone = variants[variant] ?? variants["center"]!

    context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    for row in 0..<count {
        for column in 0..<count {
            let x = Double(column) / Double(count - 1) * 2 - 1
            let y = Double(row) / Double(count - 1) * 2 - 1
            let radius = smallest + (largest - smallest) * CGFloat(min(max(tone(x, y), 0), 1))
            let center = CGPoint(x: origin + CGFloat(column) * step, y: origin + CGFloat(row) * step)
            context.fillEllipse(in: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
        }
    }
    return context.makeImage()!
}

func write(_ image: CGImage, to path: String) {
    let bitmap = NSBitmapImageRep(cgImage: image)
    guard let data = bitmap.representation(using: .png, properties: [:]) else { fatalError("no PNG for \(path)") }
    try! data.write(to: URL(fileURLWithPath: path))
}

let arguments = Array(CommandLine.arguments.dropFirst())
let chosen = "center"
if let path = arguments.first {
    let size = arguments.count > 1 ? Int(arguments[1]) ?? 1024 : 1024
    write(render(size: size, variant: arguments.count > 2 ? arguments[2] : chosen), to: path)
} else {
    for size in sizes {
        write(render(size: size, variant: chosen), to: "\(catalog)/icon_\(size).png")
    }
    print("Wrote \(sizes.count) sizes to \(catalog)")
}
