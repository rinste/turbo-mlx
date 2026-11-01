import Foundation
import MLX

/// idol-sdk's `sample_flow`: Euler steps from noise (t = 1) to the object (t = 0) on a grid shifted towards t = 1 by
/// `rescale` (`rescale_t`), with classifier-free guidance against the picture's tokens zeroed:
/// `v = (1 + w) · v(cond) − w · v(uncond)`. The noise refiner, the camera and the time run once a step for both
/// passes (they do not see the picture), each branch's picture tokens are refined once for the whole run.
public struct IdolSampler {
    /// The times in [0, 1], `steps + 1` of them, in double precision as the reference's NumPy grid.
    public let times: [Double]
    public let guidance: Float
    /// Run the unconditional pass with its picture tokens collapsed to two (`IdolDenoiser.collapsed`).
    public var collapsesUnconditional = false

    public init(steps: Int, guidance: Float = 3, rescale: Double = 3) {
        // np.linspace(1, 0, steps + 1), its last value exactly 0, then shifted.
        let linear = (0 ... steps).map { i -> Double in i == steps ? 0 : 1 + Double(i) * (-1 / Double(steps)) }
        times = linear.map { rescale * $0 / (1 + (rescale - 1) * $0) }
        self.guidance = guidance
    }

    public var steps: Int { times.count - 1 }

    /// The model's time for step `index`: `1000 · t` in double precision, then float32.
    public func modelTime(_ index: Int) -> Float { Float(1000 * times[index]) }

    /// The guided velocity of one step.
    public func velocity(_ denoiser: IdolDenoiser, latent: MLXArray, camera: MLXArray, index: Int,
                         conditional: MLXArray, unconditional: MLXArray?) -> (latent: MLXArray, camera: MLXArray) {
        let stream = denoiser.stream(latent: latent, camera: camera, t: modelTime(index))
        let cond = denoiser.joint(stream, context: conditional)
        guard guidance > 0, let unconditional else { return cond }
        let uncond = collapsesUnconditional
            ? denoiser.joint(stream, collapsed: denoiser.collapsed(unconditional))
            : denoiser.joint(stream, context: unconditional)
        let (a, b) = (1 + guidance, guidance)
        return (a * cond.latent - b * uncond.latent, a * cond.camera - b * uncond.camera)
    }

    /// The whole run from `latent` [B, 8192, 16] and `camera` [B, 1, 5] (float32 noise), conditioned on the refined
    /// picture tokens of both branches (`IdolDenoiser.context`). `progress` gets each finished step; `isCancelled` is
    /// asked between steps.
    public func run(_ denoiser: IdolDenoiser, latent: MLXArray, camera: MLXArray, conditional: MLXArray, unconditional: MLXArray?,
                    progress: (Int, Int) -> Void = { _, _ in }, isCancelled: () -> Bool = { false }) throws -> (latent: MLXArray, camera: MLXArray) {
        var (latent, camera) = (latent.asType(.float32), camera.asType(.float32))
        for index in 0 ..< steps {
            if isCancelled() { throw CancellationError() }
            let v = velocity(denoiser, latent: latent, camera: camera, index: index, conditional: conditional, unconditional: unconditional)
            let dt = Float(times[index] - times[index + 1])
            latent = latent - v.latent * dt
            camera = camera - v.camera * dt
            eval(latent, camera)
            progress(index + 1, steps)
        }
        return (latent, camera)
    }
}
