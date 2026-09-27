import AppKit
import SwiftUI

extension FocusedValues {
    /// The zoom of the image on screen, for the View menu.
    @Entry var imageZoom: ImageZoom?
}

/// Zoom level of the image viewer, shared by the viewer, its controls and the View menu.
@Observable
final class ImageZoom {
    static let maximum: CGFloat = 8
    /// Zoom In and Zoom Out move through these.
    fileprivate static let steps: [CGFloat] = [0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4, 6, 8]
    /// Offered by the percentage menu of the zoom controls.
    static let presets: [CGFloat] = [0.5, 1, 2, 4, 8]

    /// On-screen points per image pixel: 1 is actual size.
    private(set) var magnification: CGFloat = 1
    /// Shows the whole image, never enlarged past actual size.
    private(set) var fitMagnification: CGFloat = 1
    private(set) var isZoomedIn = false
    private(set) var canZoomIn = true

    @ObservationIgnored fileprivate weak var view: ZoomingScrollView?

    func zoomIn() {
        zoom(to: Self.steps.first { $0 > magnification * 1.01 } ?? Self.maximum)
    }

    func zoomOut() {
        zoom(to: Self.steps.last { $0 < magnification * 0.99 } ?? fitMagnification)
    }

    func zoomToFit() {
        zoom(to: fitMagnification)
    }

    /// Animated around the center of the view, and kept between fit and the maximum.
    func zoom(to magnification: CGFloat) {
        view?.animateZoom(to: magnification)
    }

    fileprivate func update(magnification: CGFloat, fit: CGFloat) {
        // Observation notifies on every assignment: skip unchanged values so that a pinch
        // doesn't rebuild the View menu on every frame.
        if self.magnification != magnification { self.magnification = magnification }
        if fitMagnification != fit { fitMagnification = fit }
        let isZoomedIn = magnification > fit * 1.001
        if self.isZoomedIn != isZoomedIn { self.isZoomedIn = isZoomedIn }
        let canZoomIn = magnification < Self.maximum * 0.999
        if self.canZoomIn != canZoomIn { self.canZoomIn = canZoomIn }
    }
}

/// An image fitted to the available space that zooms around the pointer: pinch, the mouse wheel
/// or ⌘-scroll; double-click to zoom in and back out; two-finger scroll (⌥-wheel with a mouse)
/// or drag to move around.
struct ZoomableImage: NSViewRepresentable {
    let url: URL
    let size: PixelSize
    let showsCheckerboard: Bool
    let zoom: ImageZoom
    /// Called on every click in the viewer, before the viewer handles it.
    var onMouseDown: () -> Void = {}

    func makeNSView(context: Context) -> ZoomingScrollView {
        let view = ZoomingScrollView(
            url: url,
            imageSize: CGSize(width: size.width, height: size.height),
            showsCheckerboard: showsCheckerboard
        )
        view.zoom = zoom
        zoom.view = view
        return view
    }

    func updateNSView(_ view: ZoomingScrollView, context: Context) {
        view.onMouseDown = onMouseDown
    }
}

/// NSScrollView does the zooming: pinch around the pointer with elastic limits, scrolling with
/// momentum, scroll bars. This adds fitting to the window, double-click, dragging to pan and
/// dragging the file out.
final class ZoomingScrollView: NSScrollView, NSDraggingSource {
    weak var zoom: ImageZoom?
    var onMouseDown: () -> Void = {}

    /// Space around a fitted image.
    private static let fitMargin: CGFloat = 28

    private let url: URL
    private let canvas: ImageCanvas
    private var fitMagnification: CGFloat = 1
    /// A fitted image follows the window size; a zoomed one keeps its magnification.
    private var isFitted = true
    private var mouseDownEvent: NSEvent?
    /// Clip view origin when a drag started panning.
    private var panStartOrigin: NSPoint?

    init(url: URL, imageSize: CGSize, showsCheckerboard: Bool) {
        self.url = url
        canvas = ImageCanvas(size: imageSize, showsCheckerboard: showsCheckerboard)
        super.init(frame: .zero)
        contentView = CenteringClipView()
        documentView = canvas
        drawsBackground = false
        contentView.drawsBackground = false
        borderType = .noBorder
        automaticallyAdjustsContentInsets = false
        hasHorizontalScroller = true
        hasVerticalScroller = true
        autohidesScrollers = true
        usesPredominantAxisScrolling = false
        allowsMagnification = true
        maxMagnification = ImageZoom.maximum
        contentView.postsBoundsChangedNotifications = true
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(clipViewBoundsDidChange),
            name: NSView.boundsDidChangeNotification,
            object: contentView
        )
        loadImage()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func loadImage() {
        if let image = ImageLoader.cached(url, maxPixelSize: nil) {
            canvas.image = image
            return
        }
        Task { [weak self, url] in
            let image = await ImageLoader.load(url, maxPixelSize: nil)
            self?.canvas.image = image
        }
    }

    // MARK: Fitting

    override func tile() {
        super.tile()
        fitToWindow()
    }

    private func fitToWindow() {
        let available = contentView.frame.size
        let image = canvas.frame.size
        guard available.width > 0, available.height > 0, image.width > 0, image.height > 0 else { return }
        let margin = 2 * Self.fitMargin
        let fit = max(0.01, min(1, (available.width - margin) / image.width, (available.height - margin) / image.height))
        fitMagnification = fit
        canvas.fitMagnification = fit
        minMagnification = fit
        if isFitted || magnification < fit {
            magnification = fit
        }
        publish()
    }

    @objc private func clipViewBoundsDidChange(_ notification: Notification) {
        isFitted = magnification <= fitMagnification * 1.001
        canvas.magnification = magnification
        let cursor: NSCursor? = canPan ? .openHand : nil
        if documentCursor != cursor { documentCursor = cursor }
        publish()
    }

    private func publish() {
        zoom?.update(magnification: magnification, fit: fitMagnification)
    }

    /// Part of the image is out of view.
    private var canPan: Bool {
        let visible = contentView.bounds.size
        let image = canvas.frame.size
        return image.width > visible.width + 0.5 || image.height > visible.height + 0.5
    }

    // MARK: Zooming

    /// Keeps `point` (image coordinates) still on screen; without one, zooms around the center.
    func animateZoom(to value: CGFloat, around point: NSPoint? = nil) {
        let target = min(max(value, fitMagnification), maxMagnification)
        let visible = contentView.bounds
        let anchor = point ?? NSPoint(x: visible.midX, y: visible.midY)
        guard !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion else {
            setMagnification(target, centeredAt: anchor)
            return
        }
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.25
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            animator().setMagnification(target, centeredAt: anchor)
        }
    }

    /// Double-click and two-finger double-tap: into the point, or back to fit.
    private func toggleZoom(at event: NSEvent) {
        if magnification > fitMagnification * 1.001 {
            animateZoom(to: fitMagnification)
        } else {
            // At least actual size and twice the fitted size, on a round percentage.
            let target = max(1, ImageZoom.steps.first { $0 >= fitMagnification * 2 } ?? ImageZoom.maximum)
            animateZoom(to: target, around: canvas.convert(event.locationInWindow, from: nil))
        }
    }

    override func smartMagnify(with event: NSEvent) {
        toggleZoom(at: event)
    }

    override func scrollWheel(with event: NSEvent) {
        let modifiers = event.modifierFlags.intersection([.command, .option])
        if event.hasPreciseScrollingDeltas {
            // Trackpad and Magic Mouse: scrolling pans; pinch, ⌘-scroll and ⌥-scroll zoom.
            guard !modifiers.isEmpty else {
                super.scrollWheel(with: event)
                return
            }
            // Once the fingers have lifted, momentum must not keep zooming.
            guard event.momentumPhase.isEmpty else { return }
            zoom(by: exp(zoomDelta(of: event) * 0.01), around: event)
        } else {
            // A mouse cannot pinch, so its wheel zooms, one step per notch; ⌥-wheel scrolls.
            guard !modifiers.contains(.option) else {
                super.scrollWheel(with: event)
                return
            }
            let notches = min(max(zoomDelta(of: event), -3), 3)
            zoom(by: pow(1.12, notches), around: event)
        }
    }

    /// Wheel away from you (or fingers up) is positive, whatever the scroll direction setting.
    private func zoomDelta(of event: NSEvent) -> CGFloat {
        let delta = event.scrollingDeltaY != 0 ? event.scrollingDeltaY : event.deltaY
        return event.isDirectionInvertedFromDevice ? -delta : delta
    }

    /// Multiplies the magnification within its limits, keeping the point under the pointer still.
    private func zoom(by factor: CGFloat, around event: NSEvent) {
        let target = min(max(magnification * factor, fitMagnification), maxMagnification)
        guard abs(target - magnification) > 0.0005 else { return }
        setMagnification(target, centeredAt: canvas.convert(event.locationInWindow, from: nil))
    }

    // MARK: Mouse

    /// Clicks anywhere in the viewer, on the image or around it, come here; scroll bars keep theirs.
    override func hitTest(_ point: NSPoint) -> NSView? {
        guard let view = super.hitTest(point) else { return nil }
        return view is NSScroller ? view : self
    }

    override var mouseDownCanMoveWindow: Bool { false }

    override func mouseDown(with event: NSEvent) {
        onMouseDown()
        if event.modifierFlags.contains(.control) {
            super.mouseDown(with: event) // control-click: the context menu
        } else if event.clickCount == 2 {
            toggleZoom(at: event)
        } else if event.clickCount == 1 {
            mouseDownEvent = event
        }
    }

    override func mouseDragged(with event: NSEvent) {
        guard let mouseDownEvent else { return }
        let start = mouseDownEvent.locationInWindow
        let location = event.locationInWindow
        if panStartOrigin == nil, canPan {
            panStartOrigin = contentView.bounds.origin
            NSCursor.closedHand.push()
        }
        if let origin = panStartOrigin {
            // The document is flipped: dragging up shows what is below.
            canvas.scroll(NSPoint(
                x: origin.x - (location.x - start.x) / magnification,
                y: origin.y + (location.y - start.y) / magnification
            ))
        } else if hypot(location.x - start.x, location.y - start.y) > 3,
                  canvas.bounds.contains(canvas.convert(start, from: nil)) {
            self.mouseDownEvent = nil
            dragFile(with: mouseDownEvent)
        }
    }

    override func mouseUp(with event: NSEvent) {
        if panStartOrigin != nil { NSCursor.pop() }
        panStartOrigin = nil
        mouseDownEvent = nil
    }

    // MARK: Dragging out

    private func dragFile(with event: NSEvent) {
        let item = NSDraggingItem(pasteboardWriter: url as NSURL)
        item.setDraggingFrame(convert(canvas.bounds, from: canvas), contents: canvas.image)
        beginDraggingSession(with: [item], event: event, source: self)
    }

    func draggingSession(_ session: NSDraggingSession, sourceOperationMaskFor context: NSDraggingContext) -> NSDragOperation {
        .copy
    }
}

/// Keeps the image centered while it is smaller than the view.
private final class CenteringClipView: NSClipView {
    override func constrainBoundsRect(_ proposedBounds: NSRect) -> NSRect {
        var rect = super.constrainBoundsRect(proposedBounds)
        guard let document = documentView?.frame else { return rect }
        if rect.width > document.width { rect.origin.x = document.midX - rect.width / 2 }
        if rect.height > document.height { rect.origin.y = document.midY - rect.height / 2 }
        return rect
    }
}

/// The document view: the image is layer content, so zooming only changes a transform.
/// Corners and shadow are scaled against the magnification to look the same at every zoom.
private final class ImageCanvas: NSView {
    var image: NSImage? {
        didSet {
            withoutAnimation {
                imageLayer.contents = image?.cgImage(forProposedRect: nil, context: nil, hints: nil)
            }
            layer?.shadowOpacity = image == nil ? 0 : 0.18
        }
    }

    var magnification: CGFloat = 1 {
        didSet { if magnification != oldValue { updateForMagnification() } }
    }

    /// Checkerboard squares are 10 points when the image is fitted.
    var fitMagnification: CGFloat = 1 {
        didSet { if fitMagnification != oldValue { updateCheckerboard() } }
    }

    private let imageLayer = CALayer()
    private let showsCheckerboard: Bool

    init(size: CGSize, showsCheckerboard: Bool) {
        self.showsCheckerboard = showsCheckerboard
        super.init(frame: CGRect(origin: .zero, size: size))
        wantsLayer = true
        layer?.shadowColor = NSColor.black.cgColor
        layer?.shadowOpacity = 0
        imageLayer.frame = bounds
        imageLayer.masksToBounds = true
        imageLayer.contentsGravity = .resize
        imageLayer.minificationFilter = .trilinear
        layer?.addSublayer(imageLayer)
        updateForMagnification()
        updateCheckerboard()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var isFlipped: Bool { true }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateCheckerboard()
    }

    private func updateForMagnification() {
        let radius = 6 / magnification
        withoutAnimation {
            imageLayer.cornerRadius = radius
            // Past 400% show the actual pixels instead of smoothing them.
            imageLayer.magnificationFilter = magnification >= 4 ? .nearest : .linear
        }
        layer?.shadowPath = CGPath(roundedRect: bounds, cornerWidth: radius, cornerHeight: radius, transform: nil)
        layer?.shadowRadius = 12 / magnification
        layer?.shadowOffset = CGSize(width: 0, height: -4 / magnification)
    }

    /// Same colors as `Checkerboard`, as a pattern that zooms with the image.
    private func updateCheckerboard() {
        guard showsCheckerboard else { return }
        let side = max(1, Int((10 / fitMagnification).rounded()))
        let isDark = effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        let scale = 2 // pixels per point, so the squares stay sharp on Retina displays
        let tileSide = side * 2 * scale
        guard let context = CGContext(
            data: nil, width: tileSide, height: tileSide, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return }
        let square = CGFloat(side * scale)
        context.setFillColor(CGColor(gray: isDark ? 0.24 : 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: tileSide, height: tileSide))
        context.setFillColor(CGColor(gray: isDark ? 0.18 : 0.88, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: square, height: square))
        context.fill(CGRect(x: square, y: square, width: square, height: square))
        guard let tile = context.makeImage() else { return }
        let pattern = NSImage(cgImage: tile, size: NSSize(width: side * 2, height: side * 2))
        withoutAnimation { imageLayer.backgroundColor = NSColor(patternImage: pattern).cgColor }
    }

    /// `imageLayer` isn't a view's layer, so Core Animation would animate its changes.
    private func withoutAnimation(_ changes: () -> Void) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        changes()
        CATransaction.commit()
    }
}

/// − 100% + in a corner of the viewer; the percentage opens a menu of zoom levels.
struct ZoomControls: View {
    let zoom: ImageZoom

    var body: some View {
        HStack(spacing: 0) {
            Button {
                zoom.zoomOut()
            } label: {
                Label("Zoom Out", systemImage: "minus")
                    .frame(width: 26, height: 24)
                    .contentShape(Rectangle())
            }
            .disabled(!zoom.isZoomedIn)
            .help("Zoom Out (⌘-)")

            Menu {
                Button("Zoom to Fit") { zoom.zoomToFit() }
                    .disabled(!zoom.isZoomedIn)
                Divider()
                ForEach(ImageZoom.presets, id: \.self) { value in
                    Button(Self.percent(value)) { zoom.zoom(to: value) }
                        .disabled(value < zoom.fitMagnification * 0.999)
                }
            } label: {
                Text(Self.percent(zoom.magnification))
                    .monospacedDigit()
                    .frame(minWidth: 44)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Zoom level")

            Button {
                zoom.zoomIn()
            } label: {
                Label("Zoom In", systemImage: "plus")
                    .frame(width: 26, height: 24)
                    .contentShape(Rectangle())
            }
            .disabled(!zoom.canZoomIn)
            .help("Zoom In (⌘+)")
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
        .font(.callout)
        .padding(.horizontal, 4)
        .background(.regularMaterial, in: Capsule())
        .overlay { Capsule().strokeBorder(Color.primary.opacity(0.1)) }
        .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
    }

    private static func percent(_ value: CGFloat) -> String {
        Double(value).formatted(.percent.precision(.fractionLength(0)))
    }
}
