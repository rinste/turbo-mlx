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

        func label(video: Bool) -> String {
            switch self {
            case .queued: "Queued"
            case .starting: "Starting…"
            case .loadingModel: "Loading the model…"
            case .encodingPrompt: "Reading the prompt…"
            case .denoising: "Generating"
            case .decoding: video ? "Decoding the video and sound…" : "Decoding the image…"
            case .encodingVideo: "Encoding the video…"
            case .saving: "Saving…"
            }
        }
    }

    let id = UUID()
    let model: ModelDescriptor
    let request: GenerationRequest
    let outputURL: URL

    var phase = Phase.queued {
        didSet { if phase == .decoding, oldValue != .decoding { decodeStartedAt = Date() } }
    }
    var step = 0
    /// The steps the engine says it will count (LTX-2 adds its refining steps to the ones asked).
    var reportedTotal: Int?
    var startedAt: Date?
    var isCancelling = false
    /// Its expected seconds step by step (TimeEstimate), set when it is queued.
    var plan: TimeEstimate.Plan?
    private(set) var denoiseStartedAt: Date?
    private(set) var lastStepAt: Date?
    private(set) var decodeStartedAt: Date?
    private(set) var secondsPerStep: Double?
    /// This run's time against the plan's, from the steps done so far.
    private(set) var pace = 1.0

    init(model: ModelDescriptor, request: GenerationRequest, outputURL: URL) {
        self.model = model
        self.request = request
        self.outputURL = outputURL
    }

    var totalSteps: Int { reportedTotal ?? request.steps }

    var statusLabel: String {
        if isCancelling { return "Stopping…" }
        // An upscale has one step and no prompt: its picture is what is read first.
        if model.family.isUpscaler, phase == .encodingPrompt { return "Reading the picture…" }
        if model.family.isUpscaler, phase == .denoising { return "Upscaling…" }
        if phase == .denoising { return "Step \(step) of \(totalSteps)" }
        return phase.label(video: model.family.media == .video)
    }

    /// Progress in 0...1, or nil while the current phase cannot be measured. Steps count by their
    /// expected time where there is a plan: a clip's last three take most of it.
    var fraction: Double? {
        switch phase {
        case .denoising:
            if let plan, plan.steps.count == totalSteps, case let total = plan.steps.reduce(0, +), total > 0 {
                return plan.steps.prefix(step).reduce(0, +) / total
            }
            return Double(step) / Double(max(totalSteps, 1))
        case .decoding, .encodingVideo, .saving: return 1
        default: return nil
        }
    }

    /// `fraction`, except for an upscale: its single step and its decode, about half of its time,
    /// would fill the bar at once, so it moves with the time the plan expects instead.
    func fraction(at now: Date) -> Double? {
        guard model.family.isUpscaler, phase == .denoising || phase == .decoding, let plan, let denoiseStartedAt,
              case let total = (plan.steps.reduce(0, +) + plan.decode) * pace, total > 0
        else { return fraction }
        return min(now.timeIntervalSince(denoiseStartedAt) / total, 0.98)
    }

    /// Seconds left at `now`: the plan's steps and decode at this run's pace, counting down within
    /// a step and through the decode. Without a plan, an image's average step.
    func secondsRemaining(at now: Date = Date()) -> Double? {
        switch phase {
        case .denoising:
            guard let plan, plan.steps.count == totalSteps else {
                guard model.family.media == .image, let secondsPerStep else { return nil }
                return Double(totalSteps - step) * secondsPerStep
            }
            let current = step < plan.steps.count ? plan.steps[step] * pace : 0
            let sinceStep = now.timeIntervalSince(lastStepAt ?? denoiseStartedAt ?? now)
            let after = plan.steps.dropFirst(step + 1).reduce(0, +) + plan.decode
            return max(current - sinceStep, 0) + after * pace
        case .decoding:
            guard let plan, let decodeStartedAt else { return nil }
            return max(plan.decode * pace - now.timeIntervalSince(decodeStartedAt), 0)
        default:
            return nil
        }
    }

    func beginDenoising() {
        phase = .denoising
        step = 0
        denoiseStartedAt = Date()
        lastStepAt = nil
    }

    func advance(to step: Int) {
        let now = Date()
        self.step = step
        lastStepAt = now
        guard let denoiseStartedAt, step > 0 else { return }
        let elapsed = now.timeIntervalSince(denoiseStartedAt)
        secondsPerStep = elapsed / Double(step)
        // After two steps the pace means something (the first one also warms up).
        if step >= 2, let plan, case let planned = plan.steps.prefix(step).reduce(0, +), planned > 0 {
            pace = min(max(elapsed / planned, 0.5), 3)
        }
    }
}
