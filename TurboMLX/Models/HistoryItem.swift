import Foundation
import UniformTypeIdentifiers

/// What a model produces, and what a history item holds.
nonisolated enum MediaKind: String, Codable, Hashable, Sendable {
    case image
    case video
}

/// A finished image (or clip) and the settings that produced it.
nonisolated struct HistoryItem: Identifiable, Hashable, Codable, Sendable {
    var id: UUID
    var createdAt: Date
    /// File name inside the history folder: a PNG, or a movie for `.video`.
    var fileName: String
    /// Absent in items saved before there were videos: an image.
    var media: MediaKind?
    /// A PNG frame of a video, for thumbnails and Quick Look.
    var posterFileName: String?
    var modelID: String
    var modelName: String
    var request: GenerationRequest
    var seconds: Double
    var peakMemory: Int64?
    /// Seconds per phase (load, encode, denoise, decode, save); older items have none.
    var timings: [String: Double]?

    var prompt: String { request.prompt }
    var size: PixelSize { request.size }
    var kind: MediaKind { media ?? .image }
}

/// How the strip arranges the history, left to right: by date, oldest first, or in the order the
/// user dragged the items into (the newer ones after it).
nonisolated enum HistoryOrder: String, CaseIterable, Identifiable, Sendable {
    case date
    case custom

    var id: Self { self }

    var title: String {
        switch self {
        case .date: "By Date"
        case .custom: "Custom Order"
        }
    }
}

extension UTType {
    /// A history item dragged inside the strip, to move it (declared in TurboMLX-Info.plist). The
    /// image or clip goes along as a file for other apps and the reference image.
    static let historyItem = UTType(exportedAs: "io.github.rinste.TurboMLX.history-item")
}
