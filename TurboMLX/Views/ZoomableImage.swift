import AppKit
import SwiftUI

extension FocusedValues {
    /// The zoom of the image on screen, for the View menu.
    @Entry var imageZoom: ImageZoom?
    /// Whether the image on screen is compared with the picture it was made from, for the View
    /// menu; nil when it has none.
    @Entry var imageComparison: Binding<Bool>?
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
/// or drag to move around. With an `original`, that picture shows left of a divider dragged across
/// the image, zoomed and moved with it.
struct ZoomableImage: NSViewRepresentable {
    let url: URL
    let size: PixelSize
    let showsCheckerboard: Bool
    let zoom: ImageZoom
    /// The picture the image was made from, to compare it with; nil shows the image alone.
    var original: URL?
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
        view.original = original
    }
}

/// NSScrollView does the zooming: pinch around the pointer with elastic limits, scrolling with
/// momentum, scroll bars. This adds fitting to the window, double-click, dragging to pan,
/// dragging the file out and the divider of a comparison.
final class ZoomingScrollView: NSScrollView, NSDraggingSource {
    weak var zoom: ImageZoom?
    var onMouseDown: () -> Void = {}
    /// The picture compared with the image, nil when not comparing.
    var original: URL? {
        didSet { if original != oldValue { loadOriginal() } }
    }

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
    /// While a drag moves the divider: how far right of it the press was, in image points.
    private var dividerGrabOffset: CGFloat?

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

    private func loadOriginal() {
        guard let original else {
            canvas.original = nil
            return
        }
        if let image = ImageLoader.cached(original, maxPixelSize: nil) {
            canvas.original = image
            return
        }
        Task { [weak self] in
            let image = await ImageLoader.load(original, maxPixelSize: nil)
            // Unless comparing was turned off while it loaded.
            guard let self, self.original == original else { return }
            canvas.original = image
        }
    }

    // MARK: Fitting

    override func tile() {
        super.tile()
        fitToWindow()
        // Resizing the window while zoomed in shows more or less of the image without moving the
        // clip view's bounds, so without a notification: the divider follows here.
        canvas.updateComparison()
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
        // The divider keeps its place in the part of the image in view.
        canvas.updateComparison()
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
        let point = canvas.convert(event.locationInWindow, from: nil)
        if event.modifierFlags.contains(.control) {
            super.mouseDown(with: event) // control-click: the context menu
        } else if canvas.dividerContains(point) {
            // A press on the divider is the divider's alone: a drag moves it, without panning
            // or dragging the file out, and a double-click there doesn't zoom. Anywhere else,
            // the mouse does what it does without a comparison.
            dividerGrabOffset = point.x - canvas.dividerX
            NSCursor.columnResize.push()
        } else if event.clickCount == 2 {
            toggleZoom(at: event)
        } else if event.clickCount == 1 {
            mouseDownEvent = event
        }
    }

    override func mouseDragged(with event: NSEvent) {
        if let dividerGrabOffset {
            canvas.moveDivider(to: canvas.convert(event.locationInWindow, from: nil).x - dividerGrabOffset)
            return
        }
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
        if panStartOrigin != nil || dividerGrabOffset != nil { NSCursor.pop() }
        panStartOrigin = nil
        dividerGrabOffset = nil
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
/// Comparing, the original covers the image left of the divider, in the same transform, so the
/// two can't drift apart while zooming or moving.
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

    /// The picture the image was made from, left of the divider; nil shows the image alone.
    var original: NSImage? {
        didSet {
            withoutAnimation {
                originalLayer.contents = original?.cgImage(forProposedRect: nil, context: nil, hints: nil)
                beforeLayer.isHidden = original == nil
            }
            divider.isHidden = original == nil
            updateComparison()
        }
    }

    /// Where the divider is across the part of the image in view, from 0 at its left to 1 at its
    /// right: zoomed in, it stays in place on the screen while the image moves under it, so the
    /// detail zoomed into always has a before and an after.
    private var split: CGFloat = 0.5 {
        didSet {
            updateComparison()
            NSAccessibility.post(element: divider, notification: .valueChanged)
        }
    }

    private let imageLayer = CALayer()
    /// Holds the original, cut off at the divider.
    private let beforeLayer = CALayer()
    private let originalLayer = CALayer()
    private let divider = ComparisonDivider()
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
        // Inside the image's layer, so its rounded corners cut the original too.
        beforeLayer.masksToBounds = true
        beforeLayer.isHidden = true
        originalLayer.contentsGravity = .resize
        originalLayer.minificationFilter = .trilinear
        beforeLayer.addSublayer(originalLayer)
        imageLayer.addSublayer(beforeLayer)
        divider.isHidden = true
        divider.adjust = { [weak self] step in
            guard let self else { return }
            split = min(max(split + step, 0), 1)
        }
        addSubview(divider)
        updateForMagnification()
        updateCheckerboard()
        updateStageGray()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var isFlipped: Bool { true }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateCheckerboard()
        updateStageGray()
    }

    // MARK: Comparing

    /// The divider's place, in image points.
    var dividerX: CGFloat {
        let visible = visiblePart
        return visible.minX + split * visible.width
    }

    /// On the divider's strip, where a press grabs it.
    func dividerContains(_ point: NSPoint) -> Bool {
        !divider.isHidden && divider.frame.contains(point)
    }

    /// Puts the divider at `x` (image points), kept within the part of the image in view.
    func moveDivider(to x: CGFloat) {
        let visible = visiblePart
        split = min(max((x - visible.minX) / visible.width, 0), 1)
    }

    /// The part of the image in view. `visibleRect` alone also takes in the margin around a
    /// fitted image, which the centering clip view shows.
    private var visiblePart: CGRect {
        let visible = visibleRect.intersection(bounds)
        return visible.width > 0 && visible.height > 0 ? visible : bounds
    }

    /// Lays out the original and the divider for the part of the image in view and the zoom.
    func updateComparison() {
        guard let original else { return }
        let x = dividerX
        let visible = visiblePart
        let frame = Self.frame(of: original.size, in: bounds)
        let width = ComparisonDivider.width / magnification
        withoutAnimation {
            beforeLayer.frame = CGRect(x: 0, y: 0, width: x, height: bounds.height)
            originalLayer.frame = frame
            // As for the image, past 400% of its own pixels it shows them.
            originalLayer.magnificationFilter = magnification * frame.width / max(original.size.width, 1) >= 4 ? .nearest : .linear
            divider.frame = CGRect(x: x - width / 2, y: visible.minY, width: width, height: visible.height)
            divider.update(magnification: magnification, value: split)
        }
        window?.invalidateCursorRects(for: divider)
    }

    /// Where the original goes in the image's frame. In the image's proportions, give or take the
    /// rounding of the sides, it fills the frame: that is how SeedVR2 reads the picture it
    /// upscales (resized to exactly the result's size), so each detail sits over its upscaled
    /// self. When an edit changed the format, the model read the picture in its own proportions;
    /// stretched to the image's, it would look distorted, and cropped it would lose what the edit
    /// may have kept, so it is fitted in the frame instead, whole and centered, the stage's gray
    /// around it.
    private static func frame(of size: CGSize, in frame: CGRect) -> CGRect {
        guard size.width > 0, size.height > 0 else { return frame }
        let scale = min(frame.width / size.width, frame.height / size.height)
        let fitted = CGSize(width: size.width * scale, height: size.height * scale)
        // Sides rounded to multiples of 16 change the proportions by about 1% at most.
        guard fitted.width < frame.width * 0.98 || fitted.height < frame.height * 0.98 else { return frame }
        return CGRect(x: frame.midX - fitted.width / 2, y: frame.midY - fitted.height / 2,
                      width: fitted.width, height: fitted.height)
    }

    /// `OutputPanel`'s stage gray, under the original: around a fitted one it reads as no picture,
    /// and the image never shows through on the before side.
    private func updateStageGray() {
        let isDark = effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        withoutAnimation { beforeLayer.backgroundColor = CGColor(gray: isDark ? 0.1 : 0.82, alpha: 1) }
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
        let checkerboard = NSColor(patternImage: pattern).cgColor
        withoutAnimation {
            imageLayer.backgroundColor = checkerboard
            originalLayer.backgroundColor = checkerboard
        }
    }

    /// `imageLayer` isn't a view's layer, so Core Animation would animate its changes.
    private func withoutAnimation(_ changes: () -> Void) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        changes()
        CATransaction.commit()
    }
}

/// The divider of a comparison: a line down the part of the image in view with a knob halfway,
/// both the same size at every zoom, on a strip as wide as the knob, where a press grabs it (the
/// scroll view takes the clicks; this only shows the cursor). For VoiceOver it is a slider, moved
/// in steps of 5% by its increment and decrement actions: the arrow keys stay with the history.
private final class ComparisonDivider: NSView {
    /// The strip and the knob, in points on the screen.
    static let width: CGFloat = 24

    /// Moves the divider by a fraction of the part of the image in view.
    var adjust: (CGFloat) -> Void = { _ in }
    private var value: CGFloat = 0.5
    private let line = CALayer()
    private let knob = CALayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        // White with a dark edge, to show on light and dark pictures alike.
        line.backgroundColor = .white
        line.shadowColor = .black
        line.shadowOpacity = 0.45
        line.shadowOffset = .zero
        knob.bounds = CGRect(x: 0, y: 0, width: Self.width, height: Self.width)
        knob.cornerRadius = Self.width / 2
        knob.backgroundColor = .white
        knob.shadowColor = .black
        knob.shadowOpacity = 0.3
        knob.shadowRadius = 3
        knob.shadowOffset = .zero
        // ‹ › across the knob: it moves sideways.
        let arrows = CAShapeLayer()
        arrows.frame = knob.bounds
        let path = CGMutablePath()
        path.addLines(between: [CGPoint(x: 10, y: 8), CGPoint(x: 6, y: 12), CGPoint(x: 10, y: 16)])
        path.addLines(between: [CGPoint(x: 14, y: 8), CGPoint(x: 18, y: 12), CGPoint(x: 14, y: 16)])
        arrows.path = path
        arrows.fillColor = nil
        arrows.strokeColor = CGColor(gray: 0.3, alpha: 1)
        arrows.lineWidth = 1.5
        arrows.lineCap = .round
        arrows.lineJoin = .round
        knob.addSublayer(arrows)
        layer?.addSublayer(line)
        layer?.addSublayer(knob)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    /// Lines up the line and the knob in the strip, after it moved or the zoom changed.
    func update(magnification: CGFloat, value: CGFloat) {
        self.value = value
        line.frame = CGRect(x: bounds.midX - 1 / magnification, y: 0, width: 2 / magnification, height: bounds.height)
        line.shadowRadius = 2 / magnification
        knob.position = CGPoint(x: bounds.midX, y: bounds.midY)
        knob.transform = CATransform3DMakeScale(1 / magnification, 1 / magnification, 1)
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .columnResize)
    }

    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .slider }
    override func accessibilityLabel() -> String? { "Divider between before and after" }
    override func accessibilityHelp() -> String? { "The original is on its left, the image on its right." }
    override func accessibilityValue() -> Any? { Double(value).formatted(.percent.precision(.fractionLength(0))) }

    override func accessibilityPerformIncrement() -> Bool {
        adjust(0.05)
        return true
    }

    override func accessibilityPerformDecrement() -> Bool {
        adjust(-0.05)
        return true
    }
}

/// − 100% + in a corner of the viewer; the percentage opens a menu of zoom levels. Before them,
/// for an image made from a picture, the switch that compares the two.
struct ZoomControls: View {
    let zoom: ImageZoom
    /// Whether the image is compared with the picture it was made from; nil when it has none.
    var comparison: Binding<Bool>?

    var body: some View {
        HStack(spacing: 0) {
            if let comparison {
                Toggle(isOn: comparison) {
                    Label("Compare with Original", systemImage: "square.split.2x1")
                        .symbolVariant(comparison.wrappedValue ? .fill : .none)
                        .frame(width: 26, height: 24)
                        .contentShape(Rectangle())
                }
                .toggleStyle(.button)
                .help(comparison.wrappedValue ? "Stop comparing (⌥⌘C)" : "Compare with the original (⌥⌘C)")

                Divider()
                    .frame(height: 14)
                    .padding(.horizontal, 2)
            }

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

/// "Before" and "After" in the top corners of the viewer while it compares, the sides of the
/// divider they name, in capsules like the zoom controls'.
struct ComparisonLabels: View {
    var body: some View {
        HStack {
            label("Before")
            Spacer()
            label("After")
        }
        .font(.callout)
        .allowsHitTesting(false)
    }

    private func label(_ title: LocalizedStringKey) -> some View {
        Text(title)
            .padding(.horizontal, 10)
            .frame(height: 24)
            .background(.regularMaterial, in: Capsule())
            .overlay { Capsule().strokeBorder(Color.primary.opacity(0.1)) }
            .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
    }
}
