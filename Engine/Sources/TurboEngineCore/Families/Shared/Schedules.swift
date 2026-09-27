import Foundation
import MLX

/// mflux's `LinearScheduler`: `steps` sigmas from 1 down to 1/steps (shifted for the image's token
/// count when the model asks for it), then 0. Z-Image and Qwen-Image run it; each step moves the
/// latents by `(σ[t+1] − σ[t]) · noise`.
public struct LinearSchedule: Equatable, Sendable {
    /// `sigma_base_shift`, `sigma_max_shift` and the sequence lengths of an mflux `ModelConfig`,
    /// with the optional `sigma_shift_terminal`.
    public struct Shift: Equatable, Sendable {
        public var baseShift: Double = 0.5
        public var maxShift: Double = 1.15
        public var baseSeqLen: Double = 256
        public var maxSeqLen: Double = 4096
        public var terminal: Double?

        public init(baseShift: Double = 0.5, maxShift: Double = 1.15, baseSeqLen: Double = 256, maxSeqLen: Double = 4096, terminal: Double? = nil) {
            self.baseShift = baseShift
            self.maxShift = maxShift
            self.baseSeqLen = baseSeqLen
            self.maxSeqLen = maxSeqLen
            self.terminal = terminal
        }

        /// The shift for an image of `width` × `height` pixels: mflux computes it from
        /// `width * height / 256`, the token count at 16 pixels per token.
        func mu(width: Int, height: Int) -> Double {
            let m = (maxShift - baseShift) / (maxSeqLen - baseSeqLen)
            let b = baseShift - m * baseSeqLen
            return m * Double(width) * Double(height) / 256 + b
        }
    }

    public let sigmas: [Float]

    public init(steps: Int, width: Int, height: Int, shift: Shift?) {
        // mx.linspace(1, 1/steps, steps) in float32.
        var sigmas = (0 ..< steps).map { i -> Float in
            steps > 1 ? 1 + Float(i) * (1 / Float(steps) - 1) / Float(steps - 1) : 1
        }
        if let shift {
            let expMu = Float(exp(shift.mu(width: width, height: height)))
            sigmas = sigmas.map { s in expMu / (expMu + (1 / s - 1)) }
            if let terminal = shift.terminal {
                let oneMinus = sigmas.map { 1 - $0 }
                let scale = oneMinus[oneMinus.count - 1] / (1 - Float(terminal))
                sigmas = oneMinus.map { 1 - $0 / scale }
            }
        }
        self.sigmas = sigmas + [0]
    }

    public var steps: Int { sigmas.count - 1 }

    /// One Euler step, in the latents' dtype: `latents + noise · (σ[t+1] − σ[t])`.
    public func step(latents: MLXArray, noise: MLXArray, index: Int) -> MLXArray {
        let dt = MLXArray(sigmas[index + 1] - sigmas[index]).asType(latents.dtype)
        return latents + noise.asType(latents.dtype) * dt
    }
}

/// Ming-Image's schedule: `FlowMatchEulerDiscreteScheduler` as the official pipeline configures
/// it, `steps` sigmas from 1 to 0 under the checkpoint's static shift of 6, plus the terminal zero.
/// The last sigma is 0, so its step is skipped (11 transformer passes for 12 steps).
public struct StaticShiftSchedule: Equatable, Sendable {
    public let sigmas: [Float]

    public init(steps: Int, shift: Float = 6) {
        // mx.linspace(1, 0, steps) in float32, then shift · s / (1 + (shift − 1) · s).
        let linear = (0 ..< steps).map { i -> Float in
            steps > 1 ? 1 - Float(i) / Float(steps - 1) : 1
        }
        sigmas = linear.map { s in shift * s / (1 + (shift - 1) * s) } + [0]
    }
}
