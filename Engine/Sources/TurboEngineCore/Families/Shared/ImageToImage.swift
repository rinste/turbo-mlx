import Foundation
import MLX

/// Image to image as mflux runs it (`Config.init_time_step`, `LatentCreator.create_for_txt2img_or_img2img`):
/// the picture, encoded by the family's VAE, is mixed with the seed's noise at the sigma of a later
/// step, and the loop runs from that step on. Z-Image, Qwen-Image and FLUX.2 Klein take it.
///
/// The strength is mflux's `image_strength`: how much of the picture the image keeps. 0.4 skips the
/// first 40% of the steps (rounded down, at least one), so the image comes out close to the picture
/// at a high strength and only takes its layout and colours at a low one. The app shows the
/// opposite, how much may change, as A1111 and InvokeAI's denoising strength do.
/// A family whose loop can start from a picture (`FamilyRequest.imageStrength`).
public protocol ImageToImageFamily: FamilyModel {}

public enum ImageToImage {
    /// The step the loop starts from: `max(1, floor(steps · strength))`, the strength clamped to
    /// [0, 1]; 0 (an image from noise alone) without a strength above 0.
    public static func startStep(steps: Int, strength: Double?) -> Int {
        guard let strength, strength > 0 else { return 0 }
        return max(1, Int(Double(steps) * min(1, strength)))
    }

    /// `add_noise_by_interpolation`: `(1 − σ) · clean + σ · noise`. mflux's sigma is a float32 array,
    /// so the mix comes out in float32 whatever the noise's precision, and the loop runs on it so.
    public static func noised(clean: MLXArray, noise: MLXArray, sigma: Float) -> MLXArray {
        let sigma = MLXArray(sigma)
        return (1 - sigma) * clean + sigma * noise
    }

    /// The picture at `path` as mflux's `LatentCreator.encode_image` hands it to a VAE: resized with
    /// Pillow's Lanczos to exactly `width` × `height` (stretched, not cropped, when the proportions
    /// differ), [1, H, W, 3] float32 in [-1, 1].
    public static func pixels(path: String, width: Int, height: Int) throws -> MLXArray {
        try QwenEditPicture(path: path).vaeInput(width: width, height: height)
    }
}
