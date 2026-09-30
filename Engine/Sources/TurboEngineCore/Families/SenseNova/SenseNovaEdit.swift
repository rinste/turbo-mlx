import Foundation
import MLX

// What SenseNova-U1.5 needs to edit a picture (`it2i_generate`): the picture resized and
// normalized as SenseTime's editing example and `load_image_native` prepare it, its tokens in the
// prompt between `<img>` and `</img>` (all at one text position, each at its row and column), and
// the query around it.

/// The picture an edit starts from, resized for the model.
public struct SenseNovaPicture {
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int

    /// ImageNet's normalization (`load_image_native`).
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]
    /// `load_image_native`'s bounds for an edit's picture.
    public static let minPixels = 512 * 512
    public static let maxPixels = 2048 * 2048

    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb
        self.width = width
        self.height = height
    }

    /// `picture` (its bytes as read, sRGB over white) resized as the reference prepares it: first
    /// to about `area` pixels in its proportions, sides multiples of `factor`, with Lanczos (the
    /// editing example's `_resize_to_max_budget`, whose budget is 2048² there and the image's area
    /// here, within 512² and 2048² as the example requires, so that the picture's tokens line up
    /// with the image's as they do in the example, where both are 2048²); then within 512² to 2048²
    /// pixels with Pillow's default bicubic (`load_image_native`, which the first step already
    /// satisfies). Each resize happens only when it changes the size, as in Pillow.
    public init(_ picture: QwenEditPicture, area: Int, factor: Int = 32) {
        var (rgb, width, height) = (picture.rgb, picture.width, picture.height)
        let budgetArea = min(max(area, Self.minPixels), Self.maxPixels)
        let budget = Self.smartResize(height: height, width: width, factor: factor, minPixels: budgetArea, maxPixels: budgetArea)
        if (budget.width, budget.height) != (width, height) {
            rgb = PILResample.resize(rgb, width: width, height: height, toWidth: budget.width, toHeight: budget.height, filter: .lanczos)
            (width, height) = (budget.width, budget.height)
        }
        let bounded = Self.smartResize(height: height, width: width, factor: factor, minPixels: Self.minPixels, maxPixels: Self.maxPixels)
        if (bounded.width, bounded.height) != (width, height) {
            rgb = PILResample.resize(rgb, width: width, height: height, toWidth: bounded.width, toHeight: bounded.height, filter: .bicubic)
            (width, height) = (bounded.width, bounded.height)
        }
        self.init(rgb: rgb, width: width, height: height)
    }

    /// `T.ToTensor()` then `T.Normalize(IMAGENET_MEAN, IMAGENET_STD)` in float32: [1, H, W, 3].
    public var pixelValues: MLXArray {
        var values = [Float](repeating: 0, count: rgb.count)
        for index in 0 ..< rgb.count {
            let channel = index % 3
            values[index] = (Float(rgb[index]) / 255 - Self.mean[channel]) / Self.std[channel]
        }
        return MLXArray(values, [1, height, width, 3])
    }

    /// The picture's tokens: one per 32 × 32 pixels.
    public func tokens(tokenSize: Int = 32) -> Int { (width / tokenSize) * (height / tokenSize) }

    /// `smart_resize` as SenseTime's `utils.py` copies it from Qwen2.5-VL: sides rounded (half to
    /// even) to multiples of `factor`, at least `factor`, the area brought within the bounds.
    public static func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int) -> (height: Int, width: Int) {
        var h = max(factor, Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        var w = max(factor, Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        if h * w > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (h, w)
    }
}

extension SenseNovaConfig {
    /// An edit's prompt as `it2i_generate` completes it for one picture: its placeholder first,
    /// unless the prompt already has one.
    public static func editPrompt(_ prompt: String) -> String {
        prompt.contains("<image>") ? prompt : "<image>\n" + prompt
    }

    /// The unconditional query of an edit's guidance (`query_img_condition`): the picture alone, no
    /// system message.
    public static let editUnconditionalQuery = "<|im_start|>user\n<image><|im_end|>\n<|im_start|>assistant\n<img>"
}

extension SenseNovaPositions {
    /// `get_thw_indexes`: each token one text position after the previous, except a picture's
    /// tokens (`imageContext`), which share the position after their `<img>`; theirs also carry the
    /// row and column in the picture's token grid (`grids`, one per picture, in order).
    public static func query(ids: [Int], imageStart: Int, imageContext: Int, grids: [(rows: Int, columns: Int)]) -> SenseNovaPositions {
        var t: [Int32] = []
        var h: [Int32] = []
        var w: [Int32] = []
        var position = -1
        var previousWasStart = false
        var picture = 0
        var inPicture = 0
        for id in ids {
            let isContext = id == imageContext
            position += (previousWasStart ? 1 : 0) + (isContext ? 0 : 1)
            t.append(Int32(position))
            if isContext, picture < grids.count {
                let columns = grids[picture].columns
                h.append(Int32(inPicture / columns))
                w.append(Int32(inPicture % columns))
                inPicture += 1
                if inPicture == grids[picture].rows * columns {
                    picture += 1
                    inPicture = 0
                }
            } else {
                h.append(0)
                w.append(0)
            }
            previousWasStart = id == imageStart
        }
        return SenseNovaPositions(t: t, h: h, w: w)
    }
}
