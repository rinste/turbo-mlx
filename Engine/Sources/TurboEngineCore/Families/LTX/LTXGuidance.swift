import Foundation
import MLX

// LTX-2's full pipeline (`TI2VidTwoStagesPipeline` in dgrauet's ltx-2-mlx, after Lightricks'
// `ti2vid_two_stages.py`): stage 1 runs the dev (undistilled) transformer at half resolution with
// multimodal guidance, on the schedule its token count shifts; stage 2 is the distilled pipeline's.
// What the guidance needs is here: the schedule, the guider and its defaults, and the negative
// prompt the reference encodes when none is given.

/// `MultiModalGuider` with constant params (`MultiModalGuiderParams`): classifier-free guidance
/// against the negative prompt, STG against a pass whose self-attention is perturbed in some
/// blocks, the modality guidance against a pass without the audio–video cross-attention, and the
/// result rescaled towards the conditioned prediction's spread.
public struct LTXGuider {
    public var cfg: Double
    public var stg: Double
    public var rescale: Double
    public var modality: Double

    /// `LTX_2_3_PARAMS` (both 2.3 and 2.5 run with them): the video's, its CFG the request's.
    public static func video(cfg: Double) -> LTXGuider { LTXGuider(cfg: cfg, stg: 1, rescale: 0.7, modality: 3) }
    /// The audio's: a stronger CFG, the same rest.
    public static let audio = LTXGuider(cfg: 7, stg: 1, rescale: 0.7, modality: 3)
    /// The blocks STG perturbs.
    public static let stgBlocks: Set<Int> = [28]

    var unconditional: Bool { abs(cfg - 1) > 1e-9 }
    var perturbed: Bool { abs(stg) > 1e-9 }
    var isolated: Bool { abs(modality - 1) > 1e-9 }

    /// `MultiModalGuider.calculate`, in the predictions' precision with the scales as weak scalars,
    /// term by term in the reference's order. A pass that did not run counts as 0.
    public func combine(cond: MLXArray, uncond: MLXArray?, perturbed: MLXArray?, isolated: MLXArray?) -> MLXArray {
        func term(_ other: MLXArray?) -> MLXArray { other.map { cond - $0 } ?? cond }
        var prediction = cond + Float(cfg - 1) * term(uncond)
        prediction = prediction + Float(stg) * term(perturbed)
        prediction = prediction + Float(modality - 1) * term(isolated)
        if rescale != 0 {
            let factor = sqrt(cond.variance()) / (sqrt(prediction.variance()) + Float(1e-8))
            prediction = prediction * (Float(rescale) * factor + Float(1 - rescale))
        }
        return prediction
    }
}

public enum LTXFullPipeline {
    /// Stage 1's steps by default (`LTX_2_3_PARAMS.num_inference_steps`).
    public static let defaultSteps = 30
    /// The video's CFG scale by default.
    public static let defaultGuidance = 3.0

    /// `DEFAULT_NEGATIVE_PROMPT`: what the reference steers away from when no negative prompt is
    /// given, the picture's flaws and the sound's.
    public static let defaultNegativePrompt =
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        + "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        + "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        + "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        + "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        + "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        + "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        + "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        + "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        + "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        + "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."

    /// `ltx2_schedule` (`dynamic_shift_schedule`): `steps + 1` sigmas from 1 to 0, shifted by an
    /// amount that grows with the token count (0.95 at 1024 tokens, 2.05 at 4096), then stretched so
    /// the last one above zero is 0.1. In double precision, as numpy computes it.
    public static func schedule(steps: Int, tokens: Int) -> [Double] {
        let (baseShift, maxShift, baseTokens, maxTokens, terminal) = (0.95, 2.05, 1024.0, 4096.0, 0.1)
        let slope = (maxShift - baseShift) / (maxTokens - baseTokens)
        let shift = Double(tokens) * slope + (baseShift - slope * baseTokens)
        let grow = Foundation.exp(shift)
        // `np.linspace(1, 0, steps + 1)`: i · (−1 / steps) + 1, the last set to 0.
        let step = -1.0 / Double(steps)
        var sigmas = (0 ... steps).map { $0 == steps ? 0 : Double($0) * step + 1 }
        sigmas = sigmas.map { $0 == 0 ? 0 : grow / (grow + (1 / $0 - 1)) }
        let nonzero = sigmas.filter { $0 != 0 }
        if let last = nonzero.last {
            let scale = (1 - last) / (1 - terminal)
            if scale != 0 { sigmas = sigmas.map { $0 == 0 ? 0 : 1 - (1 - $0) / scale } }
        }
        return sigmas
    }
}
