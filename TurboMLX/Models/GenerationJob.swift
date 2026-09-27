import Foundation
import Observation

/// One image being generated (or waiting in the queue).
@Observable
final class GenerationJob: Identifiable {
    enum Phase: Equatable {
        case queued
        case starting
        case loadingModel
        case encodingPrompt
        case denoising
        case decoding
        case encodingVideo
        case saving

        var label: String {
            switch self {
            case .queued: "Queued"
            case .starting: "Starting…"
            case .loadingModel: "Loading the model…"
            case .encodingPrompt: "Reading the prompt…"
            case .denoising: "Generating"
            case .decoding: "Decoding the image…"
            case .encodingVideo: "Encoding the video…"
            case .saving: "Saving…"
            }
        }
    }

    let id = UUID()
    let model: ModelDescriptor
    let request: GenerationRequest
    let outputURL: URL

    var phase = Phase.queued
    var step = 0
    var startedAt: Date?
    var isCancelling = false
    private(set) var denoiseStartedAt: Date?
    private(set) var secondsPerStep: Double?

    init(model: ModelDescriptor, request: GenerationRequest, outputURL: URL) {
        self.model = model
        self.request = request
        self.outputURL = outputURL
    }

    var totalSteps: Int { request.steps }

    var statusLabel: String {
        if isCancelling { return "Stopping…" }
        if phase == .denoising { return "Step \(step) of \(totalSteps)" }
        return phase.label
    }

    /// Progress in 0...1, or nil while the current phase cannot be measured.
    var fraction: Double? {
        switch phase {
        case .denoising: Double(step) / Double(max(totalSteps, 1))
        case .decoding, .encodingVideo, .saving: 1
        default: nil
        }
    }

    var estimatedSecondsRemaining: Double? {
        guard phase == .denoising, let secondsPerStep else { return nil }
        return Double(totalSteps - step) * secondsPerStep
    }

    func beginDenoising() {
        phase = .denoising
        step = 0
        denoiseStartedAt = Date()
    }

    func advance(to step: Int) {
        self.step = step
        if let denoiseStartedAt, step > 0 {
            secondsPerStep = Date().timeIntervalSince(denoiseStartedAt) / Double(step)
        }
    }
}
